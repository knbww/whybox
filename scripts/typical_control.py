#!/usr/bin/env python
"""Does the interpreter read THIS model, or what models of the task typically rely on?

A strict check of the main result (section 3.2 of the paper: scripts/seeds12_all.py,
results/seeds12_all.json). It is asked because an audit of a later experiment found that
wherever all target models rely on much the same thing, a predictor that never reads the
target model can reach the interpreter's accuracy. The world oracle of section 3.2 tests
"the model, not the world"; nothing so far tested "this model, not models in general".

PROTOCOL. Committed to the repository before this script is run, once.
---------------------------------------------------------------------------
Question.   Does the interpreter name the cause the held-out target model ITSELF relies on,
            including on the states where that model relies on something other than what
            models of the same task typically rely on?
Population. Exactly section 3.2's. Fit: initialisations 8..19 x decoy strengths 0, 1, 2
            (results/cache/volume/new_models.pt). Held out: 40..51 x 0, 1, 2
            (results/cache/fresh/held.pt). Both checked against their manifests' sha256.
            The held-out targets were scored once before, by seeds12_all.py, for the
            interpreter, world_oracle, constant and random arms only. Nothing below -- the
            typical_oracle, the data_only arm, the B_SPECIFIC cell -- was ever computed on
            them before this run.
Interpreter. seeds12_all's recipe unchanged (follows_b: prepare, tensors, train_pointer with
            its defaults, named; interpreter seeds 0, 1, 2; margin 0.10). CONTROL: its
            accuracy on every held-out target in the cells ALL, AGREE and DISAGREE must equal
            results/seeds12_all.json to 1e-9; otherwise the script stops before any new arm
            is scored.
New arms. Neither reads the held-out model.
  typical_oracle  At each held-out state, every one of the 36 fitting models names the
                  factor whose do(f := neutral) moves ITS OWN output most, measured by
                  executing the intervention on that fitting model, in units of its own
                  logit spread. The answer is the factor most of them name; a tie goes to the
                  factor with the larger mean |effect| over the 36. This is the ideal
                  predictor of what models of this task rely on, as world_oracle is the ideal
                  predictor of the world.
  data_only       A classifier that reads only the state's inputs (domain.to_model_input),
                  trained on the fitting models' margin-filtered states and their causes:
                  MLP 10-64-64-7 with ReLU, AdamW lr 1e-3, weight decay 1e-2, batch 64,
                  40 epochs, seeds 0, 1, 2; its answer is the argmax of the seed-averaged
                  probabilities. A realistic predictor of what models typically do.
  random          uniform over the seven factors, as its exact expectation 1/7.
  constant, world_oracle  as in seeds12_all.
Cells.      ALL; DISAGREE (the model's cause differs from the world's decisive factor, as in
            section 3.2); B_SPECIFIC (the model's cause differs from typical_oracle's).
            By construction world_oracle is 0 on DISAGREE and typical_oracle is 0 on
            B_SPECIFIC.
Unit.       The initialisation, 12 of them. Its value in a cell is the mean over its three
            targets that have states in that cell; an initialisation with none is reported
            and dropped from that cell for every arm.
Family.     Six paired comparisons of the interpreter, exact two-sided sign flip over the
            initialisations (mint.eval.stats.paired_sign_flip), Holm over exactly these six:
              (typical_oracle, B_SPECIFIC) (constant, B_SPECIFIC) (data_only, B_SPECIFIC)
              (random, B_SPECIFIC) (data_only, DISAGREE) (data_only, ALL)
Read-out.   SUPPORTED if (typical_oracle, B_SPECIFIC), (constant, B_SPECIFIC) and
            (data_only, B_SPECIFIC) all reject at 0.05 after Holm with a positive
            difference; NOT SUPPORTED otherwise. The ALL and DISAGREE comparisons with
            data_only are in the family and say whether those cells on their own show
            reading of the model; they do not enter the read-out.
Reported, no test. State counts per cell and target; the share of held-out states where the
            model departs from typical models; values per decoy strength.
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/typical_control.py
            It refuses to run unless this file equals the committed blob and no tracked file
            is modified, and it refuses to overwrite its result.
            `--smoke` runs the whole pipeline on fitting initialisations only (8..9 as held
            out, 10..11 as fit, 2 epochs), never touches the held-out population, and writes
            nothing.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seeds12_all as S
from follows_b import MARGIN, named, prepare, tensors, train_pointer
from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip

OUT = "results/typical_control.json"
CELLS = ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC")
FAMILY = [("typical_oracle", "B_SPECIFIC"), ("constant", "B_SPECIFIC"),
          ("data_only", "B_SPECIFIC"), ("random", "B_SPECIFIC"),
          ("data_only", "DISAGREE"), ("data_only", "ALL")]
READ_OUT = [("typical_oracle", "B_SPECIFIC"), ("constant", "B_SPECIFIC"),
            ("data_only", "B_SPECIFIC")]


def causes_of(bundle, domain, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,) the factor this model relies on at each raw state, and (N, F) the |effects|
    in units of the model's own logit spread -- executed on the model."""
    model = bundle.trained.model
    model.eval()
    run = lambda r: model(torch.as_tensor(domain.to_model_input(r)))[0].detach().double().numpy()
    with torch.no_grad():
        base = run(raw)
        eff = np.stack([run(domain.do_neutral(raw, f)) - base
                        for f in range(domain.spec.n_factors)], 1)
    a = np.abs(eff) / max(bundle.gt.probe.logit_std, 1e-9)
    return a.argmax(1), a


def typical(fit_bundles, domain, raw: np.ndarray) -> np.ndarray:
    """The factor most fitting models rely on at each state; ties by mean |effect|."""
    votes = np.zeros((len(raw), domain.spec.n_factors))
    mass = np.zeros_like(votes)
    for b in fit_bundles:
        c, a = causes_of(b, domain, raw)
        votes[np.arange(len(raw)), c] += 1
        mass += a
    best = votes.max(1, keepdims=True)
    return np.where(votes == best, mass, -np.inf).argmax(1)


class DataOnly(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def fit_data_only(x: np.ndarray, y: np.ndarray, n_out: int, epochs: int, seed: int) -> DataOnly:
    torch.manual_seed(seed)
    m = DataOnly(x.shape[1], n_out)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 64):
            j = perm[i:i + 64]
            loss = F.cross_entropy(m(xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    here = os.path.relpath(__file__)
    if not a.smoke:
        if os.path.exists(OUT):
            print(f"{OUT} already exists; this check runs once. Refusing.")
            return 2
        committed = S.git("hash-object", here) == S.git("rev-parse", f"HEAD:{here}")
        dirty = bool(S.git("status", "--porcelain", "--untracked-files=no"))
        if not committed or dirty:
            print("the protocol must be committed before the run and the tree clean. Stopping.")
            return 3

    fit_man = json.load(open(S.FIT_MANIFEST))
    fit_b, fit_sha = S.load_checked(S.FIT_FILE, fit_man["files"]["new_models"]["sha256"])
    if a.smoke:
        held_b, held_sha = fit_b, fit_sha
        fit_idx = S.indices(fit_b, (10, 11), S.FIT_BASE)
        held_idx = S.indices(fit_b, (8, 9), S.FIT_BASE)
        held_inits, epochs = (8, 9), 2
    else:
        held_man = json.load(open(S.HELD_MANIFEST))
        held_b, held_sha = S.load_checked(S.HELD_FILE, held_man["file"]["sha256"])
        fit_idx = S.indices(fit_b, S.FIT_INITS, S.FIT_BASE)
        held_idx = S.indices(held_b, S.HELD_INITS, S.HELD_BASE)
        held_inits, epochs = S.HELD_INITS, S.EPOCHS

    domain = get_domain(S.DOMAIN)
    spec = domain.spec
    fit = [prepare(fit_b[i], domain) for i in fit_idx]
    held = [prepare(held_b[i], domain) for i in held_idx]
    u_max = max(p["unit_x"].shape[1] for p in fit + held)
    tr = tensors(fit, u_max, spec, MARGIN)
    maj = int(np.bincount(tr[5], minlength=spec.n_factors).argmax())
    models = [train_pointer(tr, spec, seed=s, epochs=epochs) for s in S.SEEDS]
    print(f"interpreters trained on {len(fit)} fitting targets ({time.time() - t0:.0f}s)",
          flush=True)

    # per-state correctness of the interpreter, with section 3.2's control
    states = []
    for i, p in zip(held_idx, held):
        k = p["margin"] > MARGIN
        y = p["y"][k]
        interp = np.mean([(named(m, p, spec, u_max, MARGIN) == y) for m in models], 0)
        states.append({"i": i, "k": k, "y": y, "world": p["world"][k], "interp": interp,
                       "raw": held_b[i].focal_raw[k]})
    if not a.smoke:
        ref = json.load(open(S.OUT))["per_target"]
        for st, r in zip(states, ref):
            agree = st["world"] == st["y"]
            got = {"ALL": st["interp"].mean(), "AGREE": st["interp"][agree].mean(),
                   "DISAGREE": st["interp"][~agree].mean()}
            want = r["accuracy"]["interpreter"]
            for c, v in got.items():
                if not np.isclose(v, want[c], atol=1e-9, equal_nan=True):
                    print(f"CONTROL FAILED: target {r['initialisation']}/{r['strength']} {c}: "
                          f"{v} != {want[c]}. Stopping before any new arm is scored.")
                    return 4
        print("control passed: the interpreter reproduces section 3.2 on every target",
              flush=True)

    # the data_only arm: inputs only, trained on the fitting targets
    xs = np.concatenate([domain.to_model_input(fit_b[i].focal_raw[p["margin"] > MARGIN])
                         for i, p in zip(fit_idx, fit)])
    ys = np.concatenate([p["y"][p["margin"] > MARGIN] for p in fit])
    data_models = [fit_data_only(xs, ys, spec.n_factors, 40 if not a.smoke else 2, s)
                   for s in S.SEEDS]
    fit_bundles = [fit_b[i] for i in fit_idx]

    targets = []
    for st in states:
        raw = st["raw"]
        typ = typical(fit_bundles, domain, raw)
        with torch.no_grad():
            pr = np.mean([torch.softmax(m(torch.as_tensor(domain.to_model_input(raw))), 1).numpy()
                          for m in data_models], 0)
        y = st["y"]
        ok = {"interpreter": st["interp"],
              "typical_oracle": (typ == y).astype(float),
              "data_only": (pr.argmax(1) == y).astype(float),
              "world_oracle": (st["world"] == y).astype(float),
              "constant": (np.full(len(y), maj) == y).astype(float),
              "random": np.full(len(y), 1.0 / spec.n_factors)}
        masks = {"ALL": np.ones(len(y), bool), "AGREE": st["world"] == y,
                 "DISAGREE": st["world"] != y, "B_SPECIFIC": typ != y}
        cfg = held_b[st["i"]].trained.cfg
        targets.append({"initialisation": cfg.seed, "strength": float(cfg.spurious_strength),
                        "n_states": {c: int(m.sum()) for c, m in masks.items()},
                        "accuracy": {arm: {c: (float(v[m].mean()) if m.any() else float("nan"))
                                           for c, m in masks.items()} for arm, v in ok.items()}})
    print(f"arms scored ({time.time() - t0:.0f}s)", flush=True)

    arms = list(targets[0]["accuracy"])
    unit = {arm: {c: [] for c in CELLS} for arm in arms}
    dropped = {c: [] for c in CELLS}
    n_str = S.N_STRENGTHS
    for u, s in enumerate(held_inits):
        block = targets[u * n_str:(u + 1) * n_str]
        for c in CELLS:
            has = [t for t in block if t["n_states"][c] > 0]
            if not has:
                dropped[c].append(s)
            for arm in arms:
                unit[arm][c].append(float(np.mean([t["accuracy"][arm][c] for t in has]))
                                    if has else float("nan"))

    stats, pv = [], []
    for ref, c in FAMILY:
        ok_i = [i for i in range(len(held_inits))
                if np.isfinite(unit["interpreter"][c][i]) and np.isfinite(unit[ref][c][i])]
        x = np.array([unit["interpreter"][c][i] for i in ok_i])
        b = np.array([unit[ref][c][i] for i in ok_i])
        delta, p = paired_sign_flip(x, b)
        stats.append({"reference": ref, "cell": c, "n_units": len(x), "delta": delta,
                      "a": float(x.mean()) if len(x) else float("nan"),
                      "b": float(b.mean()) if len(b) else float("nan")})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
    by = {(r["reference"], r["cell"]): r for r in stats}
    supported = all(by[k]["reject"] and by[k]["delta"] > 0 for k in READ_OUT)
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    counts = {c: int(sum(t["n_states"][c] for t in targets)) for c in CELLS}
    print(f"\nunit: initialisation, {len(held_inits)}; states {counts}; dropped {dropped}")
    print(f"{'':16s}" + "".join(f"{c:>12s}" for c in CELLS))
    for arm in arms:
        print(f"{arm:16s}" + "".join(f"{np.nanmean(unit[arm][c]):>12.4f}" for c in CELLS))
    print(f"\n{'comparison':34s}{'cell':12s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}")
    for r in stats:
        print(f"interpreter vs {r['reference']:19s}{r['cell']:12s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}")
    print(f"\nREAD-OUT: {verdict}")
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0

    by_strength = {str(j): {arm: {c: float(np.nanmean([t["accuracy"][arm][c]
                                                       for t in targets[j::n_str]]))
                                  for c in CELLS} for arm in arms} for j in range(n_str)}
    out = {"provenance": {"commit": S.git("rev-parse", "HEAD"),
                          "protocol_commit": S.git("log", "-1", "--format=%H", "--", here),
                          "started_utc": started, "script": here,
                          "script_sha256": S.sha256_file(__file__),
                          "fit_population": {"path": S.FIT_FILE, "sha256": fit_sha},
                          "held_population": {"path": S.HELD_FILE, "sha256": held_sha},
                          "fit_initialisations": list(S.FIT_INITS),
                          "held_initialisations": list(S.HELD_INITS),
                          "unit": "initialisation", "margin": MARGIN,
                          "interpreter_seeds": list(S.SEEDS), "epochs": epochs,
                          "constant_cause": maj, "state_counts": counts,
                          "dropped_units": dropped, "torch": torch.__version__,
                          "numpy": np.__version__, "python": platform.python_version()},
           "unit_values": unit, "comparisons": stats, "per_target": targets,
           "by_strength_descriptive": by_strength, "read_out": verdict,
           "wall_seconds": time.time() - t0}
    with open(OUT, "x") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
