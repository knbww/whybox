#!/usr/bin/env python
"""Dynamic signals, deeper: how many path segments, which signal, and in which cells.

Section 3.6 of the paper found that signals measured along the path from a state to its
all-neutral reference (16 segments) raise the interpreter's accuracy. That run used four
independent seed groups. This one asks the same question on the twelve independent
initialisations of section 3.2, adds the strict cell of scripts/typical_control.py, varies
the number of path segments, and takes the dynamic signals apart.

PROTOCOL. Committed to the repository before this script is run, once.
---------------------------------------------------------------------------
Population. Section 3.2's, unchanged: fit = initialisations 8..19 x decoy strengths 0, 1, 2
            (results/cache/volume/new_models.pt), held out = 40..51 x 0, 1, 2
            (results/cache/fresh/held.pt), sha256 checked against the manifests. The held-out
            targets were scored before by seeds12_all.py and typical_control.py; no dynamic
            signal and no arm below was ever computed on them.
Recipe.     follows_b's interpreter and training unchanged (Pointer, train_pointer with 40
            epochs, AdamW 1e-3, weight decay 1e-2, batch 64), interpreter seeds 0, 1, 2,
            margin 0.10. Arms differ only in what the interpreter reads.
Signals.    mint.encoding.dynamics.path_signals along the straight path from each state to
            its all-neutral reference, K segments: per input, the integrated gradient (ig)
            and the spread of the gradient along the path (gvar); per unit, conductance
            (cond) and activation response (dact); per state, the total change of B's output
            (total). Each is expanded as section 3.6 did (signed and absolute over the state's
            maximum, tied rank, share, sign).
Arms.       current       section 3.2's interpreter (no dynamic signal)
            dyn4, dyn8, dyn16, dyn32, dyn64
                          current + all dynamic signals with K = 4, 8, 16, 32, 64
            ig16          current + ig per input + total
            gvar16        current + gvar per input + total
            units16       current + cond and dact per unit + total
CONTROL.    `current` must reproduce results/seeds12_all.json for the interpreter on every
            held-out target, cells ALL, AGREE, DISAGREE, to 1e-9; otherwise the run stops
            before anything else is reported.
Cells.      ALL; DISAGREE (the model's cause differs from the world's decisive factor);
            B_SPECIFIC (the model's cause differs from the cause most of the 36 fitting models
            rely on at that state, measured by executing their interventions -- exactly
            typical_control.py's definition).
Unit.       The initialisation (12); its value in a cell is the mean over its three targets
            with states in that cell; an initialisation with none is dropped from that cell.
Family.     Ten paired comparisons, exact two-sided sign flip over the initialisations, Holm
            over exactly these ten:
              (dyn16 vs current: ALL) (dyn16 vs current: DISAGREE)
              (dyn16 vs current: B_SPECIFIC)
              (dyn4 vs dyn16: ALL) (dyn8 vs dyn16: ALL) (dyn32 vs dyn16: ALL)
              (dyn64 vs dyn16: ALL)
              (ig16 vs current: ALL) (gvar16 vs current: ALL) (units16 vs current: ALL)
Read-out.   Each comparison is read on its own: "better" or "worse" if it rejects at 0.05
            after Holm with that sign, otherwise "no detectable change". Three questions:
            (1) do dynamic signals improve the reading -- the dyn16 vs current rows, the
            B_SPECIFIC row saying whether they help to read this model rather than models in
            general; (2) does the number of segments matter -- the four K rows; (3) which
            dynamic signal carries the gain -- the three single-signal rows.
Reported, no test. The completeness residual of the path at every K, |sum_i ig_i - (f(ref) -
            f(x))| in units of B's logit spread, mean and 95th percentile over held-out
            states; floors constant, random (exact 1/7), world_oracle, typical_oracle per cell.
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/dynamics_deep.py
            It refuses to run unless this file equals the committed blob and no tracked file
            is modified, and it refuses to overwrite its result. `--smoke` runs on fitting
            initialisations only (8..9 as held out, 10..11 as fit, K 4 and 8, 2 epochs, one
            seed), never touches the held-out population and writes nothing.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import seeds12_all as S
from follows_b import MARGIN, pos_features, prepare, recover_name, tensors, train_pointer
from mint.domains import get_domain
from mint.encoding.dynamics import path_signals
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from typical_control import typical

OUT = "results/dynamics_deep.json"
CACHE = "results/cache/dynamics_deep/packs.pt"
KS = (4, 8, 16, 32, 64)
CELLS = ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC")
CLIP = 10.0
FAMILY = [("dyn16", "current", "ALL"), ("dyn16", "current", "DISAGREE"),
          ("dyn16", "current", "B_SPECIFIC"),
          ("dyn4", "dyn16", "ALL"), ("dyn8", "dyn16", "ALL"), ("dyn32", "dyn16", "ALL"),
          ("dyn64", "dyn16", "ALL"),
          ("ig16", "current", "ALL"), ("gvar16", "current", "ALL"), ("units16", "current", "ALL")]
WORKERS, THREADS = 5, 4
_G: dict = {}


def pack(bundle, domain, ks) -> dict:
    """Everything any arm reads about one target, for every K, plus what scoring needs."""
    base = prepare(bundle, domain)
    probe = bundle.gt.probe
    out = dict(base)
    for k in ks:
        s = path_signals(bundle.trained.model, domain, bundle.focal_raw, probe.act_std,
                         probe.logit_std, k)
        out[f"ig{k}"] = pos_features(s["ig"])
        out[f"gvar{k}"] = pos_features(s["gvar"])
        out[f"units{k}"] = np.concatenate([pos_features(s["cond"]),
                                           pos_features(np.clip(s["dact"], -CLIP, CLIP))], -1)
        out[f"total{k}"] = np.stack([s["total"], np.log1p(np.abs(s["ig"]).sum(1))],
                                    1).astype(np.float32)
        out[f"resid{k}"] = np.abs(s["ig"].sum(1) / max(probe.logit_std, 1e-9) - s["total"])
    return out


def view(p: dict, arm: str) -> dict:
    ux, px, gx = [p["unit_x"]], [p["pos_x"]], [p["global_x"]]
    if arm.startswith("dyn"):
        k = int(arm[3:])
        px += [p[f"ig{k}"], p[f"gvar{k}"]]
        ux.append(p[f"units{k}"])
        gx.append(p[f"total{k}"])
    elif arm != "current":
        name, k = arm.rstrip("0123456789"), int(arm[len(arm.rstrip("0123456789")):])
        if name == "units":
            ux.append(p[f"units{k}"])
        else:
            px.append(p[f"{name}{k}"])
        gx.append(p[f"total{k}"][:, :1])
    return {"y": p["y"], "world": p["world"], "margin": p["margin"],
            "unit_x": np.concatenate(ux, -1), "pos_x": np.concatenate(px, -1),
            "global_x": np.concatenate(gx, -1)}


def _init(cache: str, threads: int) -> None:
    torch.set_num_threads(threads)
    _G["packs"] = torch.load(cache, weights_only=False)
    _G["spec"] = get_domain(S.DOMAIN).spec


def job(args):
    arm, seed, epochs = args
    spec, P = _G["spec"], _G["packs"]
    fit = [view(p, arm) for p in P["fit"]]
    held = [view(p, arm) for p in P["held"]]
    u_max = max(v["unit_x"].shape[1] for v in fit + held)
    t0 = time.time()
    m = train_pointer(tensors(fit, u_max, spec, MARGIN), spec, seed=seed, epochs=epochs)
    preds = []
    with torch.no_grad():
        for v in held:
            tr = tensors([v], u_max, spec, MARGIN)
            preds.append(recover_name(m(*tr[:4]).numpy(), spec))
    print(f"  {arm:8s} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return (arm, seed), preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    here = os.path.relpath(__file__)
    if not a.smoke:
        if os.path.exists(OUT):
            print(f"{OUT} already exists; this study runs once. Refusing.")
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
        held_inits, ks, epochs, seeds, workers = (8, 9), (4, 8), 2, (0,), 2
        arms = ("current", "dyn4", "dyn8")
        cache = "results/cache/dynamics_deep_smoke/packs.pt"
    else:
        held_man = json.load(open(S.HELD_MANIFEST))
        held_b, held_sha = S.load_checked(S.HELD_FILE, held_man["file"]["sha256"])
        fit_idx = S.indices(fit_b, S.FIT_INITS, S.FIT_BASE)
        held_idx = S.indices(held_b, S.HELD_INITS, S.HELD_BASE)
        held_inits, ks, epochs, seeds, workers = S.HELD_INITS, KS, S.EPOCHS, S.SEEDS, WORKERS
        arms = ("current",) + tuple(f"dyn{k}" for k in KS) + ("ig16", "gvar16", "units16")
        cache = CACHE

    domain = get_domain(S.DOMAIN)
    spec = domain.spec
    packs = {"fit": [pack(fit_b[i], domain, ks) for i in fit_idx],
             "held": [pack(held_b[i], domain, ks) for i in held_idx]}
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    torch.save(packs, cache)
    print(f"signals measured for {len(fit_idx) + len(held_idx)} targets at K = {ks} "
          f"({time.time() - t0:.0f}s)", flush=True)

    jobs = [(arm, s, epochs) for arm in arms for s in seeds]
    with mp.get_context("spawn").Pool(workers, initializer=_init,
                                      initargs=(cache, THREADS)) as pool:
        done = dict(pool.map(job, jobs, chunksize=1))
    print(f"interpreters trained: {len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    ys_fit = np.concatenate([p["y"][p["margin"] > MARGIN] for p in packs["fit"]])
    maj = int(np.bincount(ys_fit, minlength=spec.n_factors).argmax())
    fit_bundles = [fit_b[i] for i in fit_idx]
    targets = []
    for j, (i, p) in enumerate(zip(held_idx, packs["held"])):
        k = p["margin"] > MARGIN
        y = p["y"][k]
        typ = typical(fit_bundles, domain, held_b[i].focal_raw[k])
        ok = {arm: np.mean([done[(arm, s)][j] == y for s in seeds], 0) for arm in arms}
        ok |= {"world_oracle": (p["world"][k] == y).astype(float),
               "typical_oracle": (typ == y).astype(float),
               "constant": (np.full(len(y), maj) == y).astype(float),
               "random": np.full(len(y), 1.0 / spec.n_factors)}
        masks = {"ALL": np.ones(len(y), bool), "AGREE": p["world"][k] == y,
                 "DISAGREE": p["world"][k] != y, "B_SPECIFIC": typ != y}
        cfg = held_b[i].trained.cfg
        targets.append({"initialisation": cfg.seed, "strength": float(cfg.spurious_strength),
                        "n_states": {c: int(m.sum()) for c, m in masks.items()},
                        "accuracy": {arm: {c: (float(v[m].mean()) if m.any() else float("nan"))
                                           for c, m in masks.items()} for arm, v in ok.items()},
                        "residual": {str(kk): [float(p[f"resid{kk}"][k].mean()),
                                               float(np.percentile(p[f"resid{kk}"][k], 95))]
                                     for kk in ks}})

    if not a.smoke:
        ref = json.load(open(S.OUT))["per_target"]
        for t, r in zip(targets, ref):
            for c in ("ALL", "AGREE", "DISAGREE"):
                if not np.isclose(t["accuracy"]["current"][c], r["accuracy"]["interpreter"][c],
                                  atol=1e-9, equal_nan=True):
                    print(f"CONTROL FAILED: target {t['initialisation']}/{t['strength']} {c}: "
                          f"{t['accuracy']['current'][c]} != {r['accuracy']['interpreter'][c]}. "
                          f"Stopping before anything is reported.")
                    return 4
        print("control passed: `current` reproduces section 3.2 on every target", flush=True)

    all_arms = list(targets[0]["accuracy"])
    n_str = S.N_STRENGTHS
    unit = {arm: {c: [] for c in CELLS} for arm in all_arms}
    dropped = {c: [] for c in CELLS}
    for u, s in enumerate(held_inits):
        block = targets[u * n_str:(u + 1) * n_str]
        for c in CELLS:
            has = [t for t in block if t["n_states"][c] > 0]
            if not has:
                dropped[c].append(s)
            for arm in all_arms:
                unit[arm][c].append(float(np.mean([t["accuracy"][arm][c] for t in has]))
                                    if has else float("nan"))

    family = [f for f in FAMILY if f[0] in arms and f[1] in arms]
    stats, pv = [], []
    for arm, ref, c in family:
        ok_i = [i for i in range(len(held_inits))
                if np.isfinite(unit[arm][c][i]) and np.isfinite(unit[ref][c][i])]
        x = np.array([unit[arm][c][i] for i in ok_i])
        b = np.array([unit[ref][c][i] for i in ok_i])
        delta, p = paired_sign_flip(x, b)
        stats.append({"arm": arm, "reference": ref, "cell": c, "n_units": len(x),
                      "delta": delta, "a": float(x.mean()), "b": float(b.mean())})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
        r["reading"] = ("better" if r["reject"] and r["delta"] > 0 else
                        "worse" if r["reject"] and r["delta"] < 0 else "no detectable change")

    resid = {str(kk): [float(np.mean([t["residual"][str(kk)][0] for t in targets])),
                       float(np.mean([t["residual"][str(kk)][1] for t in targets]))] for kk in ks}
    counts = {c: int(sum(t["n_states"][c] for t in targets)) for c in CELLS}
    print(f"\nunit: initialisation, {len(held_inits)}; states {counts}; dropped {dropped}")
    print(f"{'':16s}" + "".join(f"{c:>12s}" for c in CELLS))
    for arm in all_arms:
        print(f"{arm:16s}" + "".join(f"{np.nanmean(unit[arm][c]):>12.4f}" for c in CELLS))
    print("\ncompleteness residual (mean, 95th percentile), in B's logit spread:")
    for kk, v in resid.items():
        print(f"   K = {kk:>2s}:  {v[0]:.4f}  {v[1]:.4f}")
    print(f"\n{'comparison':30s}{'cell':12s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  reading")
    for r in stats:
        print(f"{r['arm'] + ' vs ' + r['reference']:30s}{r['cell']:12s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  {r['reading']}")
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0

    out = {"provenance": {"commit": S.git("rev-parse", "HEAD"),
                          "protocol_commit": S.git("log", "-1", "--format=%H", "--", here),
                          "started_utc": started, "script": here,
                          "script_sha256": S.sha256_file(__file__),
                          "fit_population": {"path": S.FIT_FILE, "sha256": fit_sha},
                          "held_population": {"path": S.HELD_FILE, "sha256": held_sha},
                          "fit_initialisations": list(S.FIT_INITS),
                          "held_initialisations": list(S.HELD_INITS), "unit": "initialisation",
                          "margin": MARGIN, "interpreter_seeds": list(seeds), "epochs": epochs,
                          "segments": list(ks), "arms": list(arms), "constant_cause": maj,
                          "state_counts": counts, "dropped_units": dropped,
                          "workers": WORKERS, "threads": THREADS, "torch": torch.__version__,
                          "numpy": np.__version__, "python": platform.python_version()},
           "unit_values": unit, "comparisons": stats, "completeness_residual": resid,
           "per_target": targets, "wall_seconds": time.time() - t0}
    with open(OUT, "x") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
