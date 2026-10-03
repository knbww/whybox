#!/usr/bin/env python
"""The main claim on twelve independent initialisations nobody has scored before.

Why this run exists. `scripts/seeds12.py` held out one target from each of the
initialisations 24..35, and which decoy strength each contributed was a rotation
chosen by hand. An audit showed its gate depends on that choice (one of three
rotations fails; the reported one was the most favourable), and its protocol was
written immediately before the run, not committed first. Worse for any rerun:
between that run and its audit, ALL 36 targets of initialisations 24..35 have
been scored, including under the per-initialisation averaging used below
(post hoc: DISAGREE 0.5455 vs constant 0.2423). A run on them would test a
hypothesis chosen after seeing the data. So the held-out set here is new.

PROTOCOL. Committed to the repository before this script is run, once.
---------------------------------------------------------------------------
Claim.      On twelve independent held-out initialisations, the interpreter
            names the cause the target model itself relies on, including on the
            states where that cause differs from the world's.
Held out.   results/cache/fresh/held.pt, built by scripts/fresh_build.py:
            initialisations 40..51 x decoy strengths 0, 1, 2 (36 targets), from
            random stream 404, sha256 in results/fresh_manifest.json. Built,
            hashed and committed together with this protocol; nothing about these
            targets beyond their configs and their own test AUC was computed
            before this run. Architecture follows the initialisation: 4 / 4 / 4.
Fit.        results/cache/volume/new_models.pt, all 36 targets of
            initialisations 8..19 (sha256 in results/volume_manifest.json). These
            were the volume study's training pool; that is disclosed, and it does
            not matter for a fit set.
Unit.       the initialisation. Per held-out target, accuracy per cell as in
            follows_b (margin 0.10; states pooled within a target; the three
            interpreter seeds averaged per state). An initialisation's value in a
            cell is the mean over its three targets, skipping a target with no
            state in the cell. An initialisation with no state in a cell is
            reported and dropped from that cell's comparisons, for every arm at
            once. Twelve independent units; smallest attainable two-sided p 2/4096.
Recipe.     scripts/follows_b.py unchanged: prepare, tensors, train_pointer at
            its default epochs (40), AdamW 1e-3, weight decay 1e-2, batch 64,
            Pointer defaults, named; interpreter seeds 0, 1, 2. The fit set is 36
            targets, so the interpreter takes three times follows_b's optimiser
            steps; it is a different fitted interpreter from follows_b's, and
            inference is conditional on it.
Arms.       interpreter; world_oracle; constant (majority label over the
            margin-filtered states of the 36 fit targets); random (uniform over
            the causes, three draws per target from numpy default_rng(0),
            averaged, as follows_b).
Cells.      ALL, AGREE, DISAGREE (B's cause vs the world's decisive factor).
Family.     six comparisons of the interpreter against a reference, paired over
            the initialisations, exact two-sided sign-flip test
            (mint.eval.stats.paired_sign_flip), Holm over exactly these six:
              (constant, DISAGREE) (world_oracle, DISAGREE) (random, DISAGREE)
              (constant, AGREE)    (random, AGREE)          (constant, ALL)
            The world oracle is 0 on DISAGREE by construction, so that primary
            only tests "better than zero"; the substantive primary is the
            constant. The Holm floor for the smallest p is 6 x 2/4096 = 0.0029.
Read-out.   CONFIRMED if (constant, DISAGREE) and (world_oracle, DISAGREE) both
            reject at 0.05 after Holm with a positive delta. VOID if they do and
            (random, DISAGREE) does not reject with a positive delta. FAIL
            otherwise.
Reported, not in the family, no test, not part of the read-out:
            per decoy strength (`by_strength_descriptive`); per target with state
            counts (`per_target`); DISAGREE restricted to states where the world
            has a cause (`DISAGREE_WORLD_HAS_CAUSE`; on this world expected to
            equal DISAGREE); the two first-order rules (mean |.| and |signed sum|
            over carriers) as a check on the measurement instrument only,
            CLAUDE.md §0.
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/seeds12_all.py \\
                2>&1 | tee results/seeds12_all.log
            It refuses to run unless this file equals the committed blob and no
            tracked file is modified, and refuses to overwrite its result.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

import follows_b
from mint.domains import annotate, get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip

from follows_b import MARGIN, carriers_in, named, prepare, tensors, train_pointer

DOMAIN = "skirmish"
FIT_FILE, FIT_MANIFEST = "results/cache/volume/new_models.pt", "results/volume_manifest.json"
HELD_FILE, HELD_MANIFEST = "results/cache/fresh/held.pt", "results/fresh_manifest.json"
FIT_INITS, FIT_BASE = tuple(range(8, 20)), 8
HELD_INITS, HELD_BASE = tuple(range(40, 52)), 40
N_STRENGTHS = 3
SEEDS = (0, 1, 2)
RANDOM_DRAWS, RANDOM_SEED = 3, 0
OUT = "results/seeds12_all.json"
CELLS = ("ALL", "AGREE", "DISAGREE")
EXTRA = "DISAGREE_WORLD_HAS_CAUSE"
FAMILY = [("constant", "DISAGREE"), ("world_oracle", "DISAGREE"), ("random", "DISAGREE"),
          ("constant", "AGREE"), ("random", "AGREE"), ("constant", "ALL")]
EPOCHS = inspect.signature(train_pointer).parameters["epochs"].default


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def load_checked(path: str, manifest_sha: str):
    got = sha256_file(path)
    if got != manifest_sha:
        raise RuntimeError(f"{path}: sha256 {got[:16]} != manifest {manifest_sha[:16]}")
    return torch.load(path, weights_only=False), got


def indices(bundles, inits: tuple[int, ...], base: int) -> list[int]:
    """All three targets of each initialisation, checked against their configs."""
    out = []
    for s in inits:
        for j in range(N_STRENGTHS):
            idx = (s - base) * N_STRENGTHS + j
            cfg = bundles[idx].trained.cfg
            if cfg.seed != s or float(cfg.spurious_strength) != float(j):
                raise RuntimeError(f"index {idx}: ({cfg.seed}, {cfg.spurious_strength}), "
                                   f"expected ({s}, {j})")
            out.append(idx)
    return out


def per_target(p: dict, no_world: np.ndarray, models, spec, u_max, maj, rng, car) -> dict:
    """Accuracies and state counts for one held-out target, follows_b conventions."""
    k = p["margin"] > MARGIN
    y, agree = p["y"][k], p["world"][k] == p["y"][k]
    fo = p["fo"][k]
    ok = {
        "interpreter": np.mean([(named(m, p, spec, u_max, MARGIN) == y) for m in models], 0),
        "world_oracle": (p["world"][k] == y).astype(float),
        "constant": (np.full(len(y), maj) == y).astype(float),
        "random": np.mean([(rng.integers(0, spec.n_factors, len(y)) == y)
                           for _ in range(RANDOM_DRAWS)], 0),
        # instrument checks only (CLAUDE.md §0); never in a verdict
        "check_first_order_meanabs": (np.stack([np.abs(fo)[:, c].mean(1) for c in car], 1)
                                      .argmax(1) == y).astype(float),
        "check_first_order_signed": (np.stack([np.abs(fo[:, c].sum(1)) for c in car], 1)
                                     .argmax(1) == y).astype(float),
    }
    masks = {"ALL": np.ones(len(y), bool), "AGREE": agree, "DISAGREE": ~agree,
             EXTRA: ~agree & ~no_world[k]}
    return {"n_states": {c: int(m.sum()) for c, m in masks.items()},
            "accuracy": {arm: {c: (float(v[m].mean()) if m.any() else float("nan"))
                               for c, m in masks.items()} for arm, v in ok.items()}}


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if os.path.exists(OUT):
        print(f"{OUT} already exists; a confirmatory run happens once. Refusing.")
        return 2

    here = os.path.relpath(__file__)
    script_sha = sha256_file(__file__)
    committed = git("hash-object", here) == git("rev-parse", f"HEAD:{here}")
    dirty_tracked = bool(git("status", "--porcelain", "--untracked-files=no"))
    head = git("rev-parse", "HEAD")
    protocol_commit = git("log", "-1", "--format=%H", "--", here)
    protocol_date = git("log", "-1", "--format=%cI", "--", here)
    print(f"{here}: sha256 {script_sha[:16]}, закоммичен: {committed}, "
          f"изменённые отслеживаемые файлы: {dirty_tracked}; протокол {protocol_commit[:8]} "
          f"от {protocol_date}; запуск {started}")
    if not committed or dirty_tracked:
        print("протокол должен быть закоммичен до запуска, а дерево чистым. Остановка.")
        return 3

    fit_man, held_man = json.load(open(FIT_MANIFEST)), json.load(open(HELD_MANIFEST))
    if held_man.get("scored_before_confirmatory_run") is not False:
        raise RuntimeError(f"{HELD_MANIFEST} does not certify an unscored held-out set")
    fit_b, fit_sha = load_checked(FIT_FILE, fit_man["files"]["new_models"]["sha256"])
    held_b, held_sha = load_checked(HELD_FILE, held_man["file"]["sha256"])
    fit_idx = indices(fit_b, FIT_INITS, FIT_BASE)
    held_idx = indices(held_b, HELD_INITS, HELD_BASE)
    describe = lambda bs, idx: [{"initialisation": bs[i].trained.cfg.seed,
                                 "strength": float(bs[i].trained.cfg.spurious_strength),
                                 "widths": list(bs[i].trained.cfg.arch.get("widths", ()))}
                                for i in idx]

    domain = get_domain(DOMAIN)
    spec = domain.spec
    fit = [prepare(fit_b[i], domain) for i in fit_idx]
    held = [prepare(held_b[i], domain) for i in held_idx]
    u_max = max(p["unit_x"].shape[1] for p in fit + held)
    tr = tensors(fit, u_max, spec, MARGIN)
    maj = int(np.bincount(tr[5], minlength=spec.n_factors).argmax())
    print(f"обучение: {len(fit)} моделей ({FIT_INITS[0]}..{FIT_INITS[-1]}), проверка: "
          f"{len(held)} моделей ({HELD_INITS[0]}..{HELD_INITS[-1]}), эпох {EPOCHS}; "
          f"константа = причина {maj} ({time.time() - t0:.0f}s)", flush=True)

    models = []
    for s in SEEDS:
        models.append(train_pointer(tr, spec, seed=s))
        print(f"    seed {s} обучен ({time.time() - t0:.0f}s)", flush=True)

    car = carriers_in(spec, held[0]["pos_x"].shape[1])
    rng = np.random.default_rng(RANDOM_SEED)
    targets = []
    for i, p in zip(held_idx, held):
        none = np.abs(annotate(domain, held_b[i].focal_raw).factor_effect).max(1) == 0
        targets.append(per_target(p, none, models, spec, u_max, maj, rng, car))

    arms = list(targets[0]["accuracy"])
    all_cells = CELLS + (EXTRA,)
    unit = {arm: {c: [] for c in all_cells} for arm in arms}
    dropped = {c: [] for c in all_cells}
    for u, s in enumerate(HELD_INITS):
        block = targets[u * N_STRENGTHS:(u + 1) * N_STRENGTHS]
        for c in all_cells:
            has = [t for t in block if t["n_states"][c] > 0]
            if not has:
                dropped[c].append(s)
            for arm in arms:
                unit[arm][c].append(float(np.mean([t["accuracy"][arm][c] for t in has]))
                                    if has else float("nan"))

    rows = []
    for arm in arms:
        for c in all_cells:
            v = unit[arm][c]
            rows.append({"method": arm, "cell": c, "unit": "initialisation",
                         "role": "reported" if c == EXTRA or arm.startswith("check_") else "arm",
                         "n_units": int(np.isfinite(v).sum()),
                         "per_model": [{"initialisation": s, "accuracy": x}
                                       for s, x in zip(HELD_INITS, v) if np.isfinite(x)],
                         "accuracy": float(np.nanmean(v))})

    stats, pv = [], []
    for ref, c in FAMILY:
        ok = [i for i in range(len(HELD_INITS))
              if np.isfinite(unit["interpreter"][c][i]) and np.isfinite(unit[ref][c][i])]
        a = np.array([unit["interpreter"][c][i] for i in ok])
        b = np.array([unit[ref][c][i] for i in ok])
        delta, p = paired_sign_flip(a, b)
        stats.append({"metric": "accuracy", "method": "interpreter", "reference": ref,
                      "cell": c, "n_models": len(a), "unit": "initialisation", "delta": delta,
                      "a": float(a.mean()), "b": float(b.mean())})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)

    positive = lambda r: r["reject"] and r["delta"] > 0
    prim = [r for r in stats if r["cell"] == "DISAGREE"
            and r["reference"] in ("constant", "world_oracle")]
    rand = [r for r in stats if r["cell"] == "DISAGREE" and r["reference"] == "random"]
    ok = len(prim) == 2 and all(positive(r) for r in prim)
    void = ok and not all(positive(r) for r in rand)
    gate = "VOID" if void else "CONFIRMED" if ok else "FAIL"

    n_units = len(HELD_INITS)
    by_strength = {"role": "reported only; not in the family; no test",
                   "strengths": {}}
    for j in range(N_STRENGTHS):
        block = targets[j::N_STRENGTHS]
        by_strength["strengths"][str(j)] = {
            arm: {c: float(np.nanmean([t["accuracy"][arm][c] for t in block])) for c in CELLS}
            for arm in ("interpreter", "constant", "world_oracle", "random")}
    per_t = [{**meta, **t} for meta, t in zip(describe(held_b, held_idx), targets)]
    counts = {c: int(sum(t["n_states"][c] for t in targets)) for c in all_cells}

    print(f"\nединица анализа: инициализация, {n_units} независимых; состояний: {counts}; "
          f"отброшено: {dropped}")
    by = {(r["method"], r["cell"]): r for r in rows}
    for arm in ("interpreter", "world_oracle", "constant", "random"):
        print(f"   {arm:14s}" + "  ".join(f"{c} {by[(arm, c)]['accuracy']:.4f}" for c in all_cells))
    print(f"\n{'comparison':34s}{'cell':10s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  sig")
    for r in stats:
        print(f"interpreter vs {r['reference']:19s}{r['cell']:10s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  "
              f"{('+' if r['delta'] > 0 else '-') if r['reject'] else ''}")
    print(f"\nGATE: {gate}   (наименьшее достижимое p: {2 / 2 ** n_units:.6f})")

    out = {"provenance": {"commit": head, "dirty_tracked": dirty_tracked,
                          "protocol_commit": protocol_commit, "protocol_commit_date": protocol_date,
                          "started_utc": started,
                          "script": here, "script_sha256": script_sha,
                          "modules": {"follows_b": os.path.relpath(follows_b.__file__),
                                      "mint": os.path.relpath(sys.modules["mint"].__file__)},
                          "protocol": "module docstring, committed before the run",
                          "domain": DOMAIN,
                          "fit_population": {"path": FIT_FILE, "sha256": fit_sha,
                                             "targets": describe(fit_b, fit_idx)},
                          "held_population": {"path": HELD_FILE, "sha256": held_sha,
                                              "targets": describe(held_b, held_idx)},
                          "fit_initialisations": list(FIT_INITS),
                          "held_initialisations": list(HELD_INITS),
                          "unit": "initialisation", "independent_units": n_units,
                          "margin": MARGIN, "interpreter_seeds": list(SEEDS),
                          "epochs": EPOCHS, "random_draws": RANDOM_DRAWS,
                          "random_seed": RANDOM_SEED, "constant_cause": maj,
                          "state_counts": counts, "dropped_units": dropped,
                          "torch": torch.__version__, "numpy": np.__version__,
                          "python": platform.python_version()},
           "rows": rows, "comparisons": stats, "per_target": per_t,
           "by_strength_descriptive": by_strength,
           "gate": gate, "wall_seconds": time.time() - t0}
    with open(OUT, "x") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
