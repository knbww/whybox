#!/usr/bin/env python
"""Car Evaluation through the whole study: the paper's method on one external dataset.

Every result of the paper so far was measured on simulated worlds. This run takes one
real, published dataset and passes it through each part of the study with the paper's
method unchanged: the main result, the strict check against what models typically rely
on, the transfer between worlds in both directions, and the dynamic signals.

PROTOCOL. Committed to the repository before this script is run, once.
---------------------------------------------------------------------------
World.      UCI Car Evaluation (Bohanec & Rajkovic, 1988), used as published:
            mint.domains.car. 1728 rows, every combination of six attributes, each
            labelled by the published hierarchical model, so the world's answer to any
            suppression is a row of the table. No column added, no label changed. The
            outcome is the published level read as a number: unacc 0, acc 1, good 2,
            vgood 3.
Causes.     The concepts of the published hierarchy: PRICE (buying, maint), COMFORT
            (doors, persons, lug_boot), SAFETY (safety). Neutral values are the middle
            values: med, med, 4 doors, 4 persons, med boot, med safety.
Targets.    scripts/car_build.py, sha256 in results/car_manifest.json, built, hashed and
            committed together with this protocol. B answers the level as a number
            (squared error, 300 epochs, AdamW 3e-3, weight decay 1e-4, batch 64) from a
            random 70% of the rows (1210) drawn per initialisation; three architectures per
            initialisation (the paper's MLP presets). fit = initialisations 0..11 (36
            targets, 160 focal rows each); held = 100..111 (36 targets, all 1728 rows
            focal). Reverse transfer: skirmish_held = the simulated tactical world,
            initialisations 60..71 x decoy strengths 0, 1, 2 (36 targets, scripts/
            fresh_build.py's recipe, stream 606). The builder stores every measurement of
            these targets in their bundles; nothing beyond their own test accuracy (Car) or
            AUC (skirmish) was printed or read before this run.
Pilot.      Disclosed: initialisations 900..902 (stream 9009, used for nothing else) fixed
            the epoch budget (test accuracy 0.948 at 100 epochs, 0.968 at 300; majority
            0.69) and showed the cells' sizes: B departs from the published rule on 0-3% of
            its margin-kept states and from the other models on 2-16%. No interpreter was
            trained or scored in the pilot.
Labels.     B's cause: follows_b unchanged -- argmax over causes of |B's response to
            do(cause := neutral)| in units of B's own output spread on its probe states;
            states kept if the top two differ by more than 0.10. The world's cause: the
            cause whose suppression changes the published level most, when exactly one does
            (mint.domains.annotate margin > 0); otherwise the world names no single cause.
Interpreters. follows_b's Pointer and train_pointer unchanged (40 epochs, AdamW 1e-3, weight
            decay 1e-2, batch 64, margin 0.10), interpreter seeds 0, 1, 2, per-state
            correctness averaged over the seeds.
  car       fitted on the 36 Car fit targets, section 3.2's signals.
  car_dyn16 the same with the dynamic signals of section 3.6 at K = 16 (scripts/
            dynamics_deep.py `dyn16` view).
  sim       fitted only on the simulated tactical world: section 3.2's interpreter, its 36
            fit targets (results/cache/volume/new_models.pt, initialisations 8..19). It
            never sees Car. CONTROL: on section 3.2's held-out targets it must reproduce
            results/seeds12_all.json for the interpreter on every target, cells ALL, AGREE,
            DISAGREE, to 1e-9; otherwise the run stops before any arm is scored.
Arms, Car held-out targets. car, car_dyn16, sim; world_oracle (the world's cause, wrong
            where the world names none); typical_oracle (the cause most of the 36 Car fit
            targets rely on at that state, executed on each; ties by mean |effect| --
            scripts/typical_control.py); data_only (MLP 6-64-64-3 on the inputs alone,
            trained on the Car fit targets' kept states and causes, 40 epochs, seeds 0, 1,
            2, typical_control's recipe); constant (the majority cause of the Car fit
            targets' kept states); random (exact 1/3).
Arms, skirmish held-out targets. car (never fitted on skirmish); sim (the in-world
            interpreter, reported); world_oracle; typical_oracle (the 36 skirmish fit
            targets); constant (the majority cause of the skirmish fit targets' kept
            states, as section 3.2); random (exact 1/7).
Cells.      ALL (kept states); AGREE and DISAGREE (B's cause equals / differs from the
            world's, among states where the world names a single cause); B_SPECIFIC (B's
            cause differs from typical_oracle's). Reported only: WORLD_SILENT (the world
            names no single cause); UNSEEN (Car rows outside B's training split).
Unit.       The initialisation, 12 per population; its value in a cell is the mean over
            its three targets with states in that cell; an initialisation with none is
            reported and dropped from that cell for every arm.
Family.     Fourteen paired comparisons, exact two-sided sign flip over the
            initialisations (mint.eval.stats.paired_sign_flip), Holm over exactly these
            fourteen; a comparison with no unit enters with p = 1.
              Car, interpreter car:
                 1 (car vs constant, ALL)
                 2 (car vs data_only, ALL)
                 3 (car vs constant, DISAGREE)
                 4 (car vs world_oracle, DISAGREE)
                 5 (car vs typical_oracle, B_SPECIFIC)
                 6 (car vs constant, B_SPECIFIC)
                 7 (car vs data_only, B_SPECIFIC)
              Car, interpreter sim (never fitted on Car):
                 8 (sim vs constant, ALL)
                 9 (sim vs constant, DISAGREE)
                10 (sim vs constant, B_SPECIFIC)
              skirmish, interpreter car (never fitted on skirmish):
                11 (car vs constant, ALL)
                12 (car vs constant, DISAGREE)
                13 (car vs constant, B_SPECIFIC)
              Car, dynamic signals:
                14 (car_dyn16 vs car, ALL)
Read-out.   Each question on its own. SUPPORTED if every listed comparison rejects at 0.05
            after Holm with a positive difference; otherwise NOT SUPPORTED, except NOT
            TESTABLE when a listed comparison that failed has fewer than 10 initialisations
            (its smallest attainable p cannot pass the first Holm step).
              Q1 the interpreter names the cause B relies on, on real data:        1
              Q2 ... where B departs from the published rule:                      3, 4
              Q3 ... where B departs from what models of the task rely on:         5, 6, 7
              Q4 trained only on the simulation, it reads models of real data:     8, 10
              Q5 trained only on real data, it reads models of the simulation:     11, 12, 13
              Q6 dynamic signals: "better" / "worse" if 14 rejects with that sign,
                 otherwise "no detectable change".
            Comparisons 2 and 9 are in the family and enter no read-out: 2 says whether
            the pooled accuracy on its own shows reading of the model; 9 rests on the
            small DISAGREE cell of a model that never saw Car.
Reported, no test. State counts per cell and target; the distribution of B's causes; the
            completeness residual of the K = 16 path, |sum_i ig_i - (f(ref) - f(x))| in
            units of B's output spread; the two first-order rules on Car, as a check on the
            measurement instrument only (CLAUDE.md section 0).
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/car_study.py
            It refuses to run unless this file equals the committed blob and no tracked
            file is modified, and it refuses to overwrite its result. `--smoke` runs the
            whole pipeline on results/cache/car_smoke (scripts/car_build.py --smoke) and
            fitting initialisations of skirmish only, 2 epochs, one seed, no control, and
            writes nothing.
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
from dynamics_deep import pack, view
from follows_b import MARGIN, carriers_in, prepare, recover_name, tensors, train_pointer
from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from typical_control import fit_data_only, typical

OUT = "results/car_study.json"
MANIFEST, SMOKE_MANIFEST = "results/car_manifest.json", "results/cache/car_smoke/manifest.json"
CACHE = "results/cache/car_study/packs.pt"
CELLS = ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC")
REPORTED = ("WORLD_SILENT", "UNSEEN")
FAMILY = [("car", "car", "constant", "ALL"), ("car", "car", "data_only", "ALL"),
          ("car", "car", "constant", "DISAGREE"), ("car", "car", "world_oracle", "DISAGREE"),
          ("car", "car", "typical_oracle", "B_SPECIFIC"),
          ("car", "car", "constant", "B_SPECIFIC"), ("car", "car", "data_only", "B_SPECIFIC"),
          ("car", "sim", "constant", "ALL"), ("car", "sim", "constant", "DISAGREE"),
          ("car", "sim", "constant", "B_SPECIFIC"),
          ("skirmish", "car", "constant", "ALL"), ("skirmish", "car", "constant", "DISAGREE"),
          ("skirmish", "car", "constant", "B_SPECIFIC"),
          ("car", "car_dyn16", "car", "ALL")]
QUESTIONS = {"Q1": (1,), "Q2": (3, 4), "Q3": (5, 6, 7), "Q4": (8, 10), "Q5": (11, 12, 13)}
DYN = 14
MIN_UNITS = 10
INTERPRETERS = ("car", "car_dyn16", "sim")
WORKERS, THREADS = 5, 4
_G: dict = {}


def _init(cache: str, threads: int, u_max: int) -> None:
    torch.set_num_threads(threads)
    _G["packs"] = torch.load(cache, weights_only=False)
    _G["u_max"] = u_max
    _G["car"], _G["sk"] = get_domain("car").spec, get_domain("skirmish").spec


def job(args):
    """Fit one interpreter and name a cause at every kept state of every target it reads."""
    name, seed, epochs = args
    P, u_max = _G["packs"], _G["u_max"]
    car, sk = _G["car"], _G["sk"]
    arm = "dyn16" if name == "car_dyn16" else "current"
    fit_packs, fit_spec = (P["sk_fit"], sk) if name == "sim" else (P["car_fit"], car)
    reads = {"car_held": car}
    if name in ("car", "sim"):
        reads["sk_held"] = sk
    if name == "sim" and P["sk_32"]:
        reads["sk_32"] = sk
    t0 = time.time()
    m = train_pointer(tensors([view(p, arm) for p in fit_packs], u_max, fit_spec, MARGIN),
                      fit_spec, seed=seed, epochs=epochs)
    preds = {}
    with torch.no_grad():
        for key, spec in reads.items():
            out = []
            for p in P[key]:
                tr = tensors([view(p, arm)], u_max, spec, MARGIN)
                out.append(recover_name(m(*tr[:4]).numpy(), spec))
            preds[key] = out
    print(f"  {name:9s} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return (name, seed), preds


def load_pt(path: str, want: str):
    got = S.sha256_file(path)
    if got != want:
        raise RuntimeError(f"{path}: sha256 {got[:16]} != manifest {want[:16]}")
    return torch.load(path, weights_only=False)


def with_world(p: dict, bundle) -> dict:
    p["world_named"] = bundle.gt.decisive_margin > 0
    return p


def scored(y, arms: dict, masks: dict) -> dict:
    return {"n_states": {c: int(m.sum()) for c, m in masks.items()},
            "accuracy": {a: {c: (float(np.asarray(v, float)[m].mean()) if m.any() else float("nan"))
                             for c, m in masks.items()} for a, v in arms.items()}}


def units(targets: list[dict], cells) -> tuple[dict, dict, list]:
    """Per initialisation: the mean over its targets with states in the cell."""
    inits = sorted({t["initialisation"] for t in targets})
    arms = list(targets[0]["accuracy"])
    val = {a: {c: [] for c in cells} for a in arms}
    dropped = {c: [] for c in cells}
    for s in inits:
        block = [t for t in targets if t["initialisation"] == s]
        for c in cells:
            has = [t for t in block if t["n_states"][c] > 0]
            if not has:
                dropped[c].append(s)
            for a in arms:
                val[a][c].append(float(np.mean([t["accuracy"][a][c] for t in has]))
                                 if has else float("nan"))
    return val, dropped, inits


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
            print(f"{OUT} already exists; this run happens once. Refusing.")
            return 2
        committed = S.git("hash-object", here) == S.git("rev-parse", f"HEAD:{here}")
        dirty = bool(S.git("status", "--porcelain", "--untracked-files=no"))
        print(f"{here}: committed {committed}, modified tracked files {dirty}; started {started}")
        if not committed or dirty:
            print("the protocol must be committed before the run and the tree clean. Stopping.")
            return 3
    seeds, epochs = ((0,), 2) if a.smoke else (S.SEEDS, S.EPOCHS)

    car, sk = get_domain("car"), get_domain("skirmish")
    man = json.load(open(SMOKE_MANIFEST if a.smoke else MANIFEST))
    if man.get("scored_before_confirmatory_run") is not False:
        raise RuntimeError("the Car manifest does not certify unscored populations")
    F = man["files"]
    car_fit = load_pt(F["fit"]["path"], F["fit"]["sha256"])
    car_held = load_pt(F["held"]["path"], F["held"]["sha256"])
    sk_held_b = load_pt(F["skirmish_held"]["path"], F["skirmish_held"]["sha256"])
    fit_man = json.load(open(S.FIT_MANIFEST))
    sk_fit_all, sk_fit_sha = S.load_checked(S.FIT_FILE, fit_man["files"]["new_models"]["sha256"])
    sk_fit_idx = S.indices(sk_fit_all, (10, 11) if a.smoke else S.FIT_INITS, S.FIT_BASE)
    sk_fit_b = [sk_fit_all[i] for i in sk_fit_idx]
    sk_32_b, sk_32_sha = [], None
    if not a.smoke:
        held_man = json.load(open(S.HELD_MANIFEST))
        all32, sk_32_sha = S.load_checked(S.HELD_FILE, held_man["file"]["sha256"])
        sk_32_b = [all32[i] for i in S.indices(all32, S.HELD_INITS, S.HELD_BASE)]
    for b in car_held["bundles"]:
        if not np.array_equal(b.focal_raw, car.states):
            raise RuntimeError("a Car held-out target is not focal on every row in table order")

    packs = {"car_fit": [with_world(pack(b, car, (16,)), b) for b in car_fit["bundles"]],
             "car_held": [with_world(pack(b, car, (16,)), b) for b in car_held["bundles"]],
             "sk_fit": [prepare(b, sk) for b in sk_fit_b],
             "sk_held": [with_world(prepare(b, sk), b) for b in sk_held_b],
             "sk_32": [prepare(b, sk) for b in sk_32_b]}
    u_max = max(p["unit_x"].shape[1] for v in packs.values() for p in v)
    if sk_32_b and u_max != max(p["unit_x"].shape[1] for p in packs["sk_fit"] + packs["sk_32"]):
        raise RuntimeError("padding differs from section 3.2's; the control would not hold")
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(packs, CACHE)
    print(f"signals measured: Car {len(packs['car_fit'])} fit / {len(packs['car_held'])} held, "
          f"skirmish {len(packs['sk_fit'])} fit / {len(packs['sk_held'])} held "
          f"({time.time() - t0:.0f}s)", flush=True)

    jobs = [(n, s, epochs) for n in INTERPRETERS for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(WORKERS, len(jobs)), initializer=_init,
                  initargs=(CACHE, THREADS, u_max)) as pool:
        preds = dict(pool.map(job, jobs))
    print(f"interpreters fitted ({time.time() - t0:.0f}s)", flush=True)
    correct = lambda name, key, t, y: np.mean([preds[(name, s)][key][t] == y for s in seeds], 0)

    # CONTROL: `sim` is section 3.2's interpreter
    if not a.smoke:
        ref = json.load(open(S.OUT))["per_target"]
        for t, (p, r) in enumerate(zip(packs["sk_32"], ref)):
            k = p["margin"] > MARGIN
            y, agree = p["y"][k], p["world"][k] == p["y"][k]
            ok = correct("sim", "sk_32", t, y)
            got = {"ALL": ok.mean(), "AGREE": ok[agree].mean() if agree.any() else np.nan,
                   "DISAGREE": ok[~agree].mean() if (~agree).any() else np.nan}
            for c, v in got.items():
                if not np.isclose(v, r["accuracy"]["interpreter"][c], atol=1e-9, equal_nan=True):
                    print(f"CONTROL FAILED: {r['initialisation']}/{r['strength']} {c}: {v} != "
                          f"{r['accuracy']['interpreter'][c]}. Stopping before any arm is scored.")
                    return 4
        print("control passed: `sim` reproduces section 3.2 on every target", flush=True)

    # Car: typical models, data-only classifier, constant
    rows_all = car.states
    typ_car = typical(car_fit["bundles"], car, rows_all)
    xs = np.concatenate([car.to_model_input(b.focal_raw[p["margin"] > MARGIN])
                         for b, p in zip(car_fit["bundles"], packs["car_fit"])])
    ys = np.concatenate([p["y"][p["margin"] > MARGIN] for p in packs["car_fit"]])
    data_models = [fit_data_only(xs, ys, car.spec.n_factors, 40 if not a.smoke else 2, s)
                   for s in S.SEEDS]
    with torch.no_grad():
        prob = np.mean([torch.softmax(m(torch.as_tensor(car.to_model_input(rows_all))), 1).numpy()
                        for m in data_models], 0)
    data_car = prob.argmax(1)
    maj_car = int(np.bincount(ys, minlength=car.spec.n_factors).argmax())
    car_rules = carriers_in(car.spec, car.spec.n_inputs)

    car_targets = []
    for t, (b, p, rec) in enumerate(zip(car_held["bundles"], packs["car_held"],
                                        car_held["records"])):
        k = p["margin"] > MARGIN
        rows = np.flatnonzero(k)
        y, world, named = p["y"][k], p["world"][k], p["world_named"][k]
        fo = p["fo"][k]
        arms = {n: correct(n, "car_held", t, y) for n in INTERPRETERS}
        arms |= {"world_oracle": np.where(named, world, -1) == y,
                 "typical_oracle": typ_car[rows] == y, "data_only": data_car[rows] == y,
                 "constant": np.full(len(y), maj_car) == y,
                 "random": np.full(len(y), 1.0 / car.spec.n_factors),
                 "check_first_order_meanabs": np.stack(
                     [np.abs(fo)[:, c].mean(1) for c in car_rules], 1).argmax(1) == y,
                 "check_first_order_signed": np.stack(
                     [np.abs(fo[:, c].sum(1)) for c in car_rules], 1).argmax(1) == y}
        masks = {"ALL": np.ones(len(y), bool), "AGREE": named & (world == y),
                 "DISAGREE": named & (world != y), "B_SPECIFIC": typ_car[rows] != y,
                 "WORLD_SILENT": ~named, "UNSEEN": ~np.isin(rows, rec["train_rows"])}
        car_targets.append({"initialisation": rec["initialisation"], "arch": rec["arch"],
                            "test_accuracy": rec["test_accuracy"],
                            "causes": np.bincount(y, minlength=3).tolist(),
                            "completeness_k16": [float(p["resid16"][k].mean()),
                                                 float(np.percentile(p["resid16"][k], 95))],
                            **scored(y, arms, masks)})

    # skirmish: the reverse transfer
    maj_sk = int(np.bincount(np.concatenate([p["y"][p["margin"] > MARGIN] for p in packs["sk_fit"]]),
                             minlength=sk.spec.n_factors).argmax())
    sk_targets = []
    for t, (b, p) in enumerate(zip(sk_held_b, packs["sk_held"])):
        k = p["margin"] > MARGIN
        y, world, named = p["y"][k], p["world"][k], p["world_named"][k]
        typ = typical(sk_fit_b, sk, b.focal_raw[k])
        arms = {n: correct(n, "sk_held", t, y) for n in ("car", "sim")}
        arms |= {"world_oracle": np.where(named, world, -1) == y, "typical_oracle": typ == y,
                 "constant": np.full(len(y), maj_sk) == y,
                 "random": np.full(len(y), 1.0 / sk.spec.n_factors)}
        masks = {"ALL": np.ones(len(y), bool), "AGREE": named & (world == y),
                 "DISAGREE": named & (world != y), "B_SPECIFIC": typ != y,
                 "WORLD_SILENT": ~named}
        cfg = b.trained.cfg
        sk_targets.append({"initialisation": cfg.seed, "strength": float(cfg.spurious_strength),
                           "causes": np.bincount(y, minlength=sk.spec.n_factors).tolist(),
                           **scored(y, arms, masks)})
    print(f"arms scored ({time.time() - t0:.0f}s)", flush=True)

    pops = {"car": car_targets, "skirmish": sk_targets}
    unit, dropped, inits = {}, {}, {}
    for name, targets in pops.items():
        cells = CELLS + REPORTED if name == "car" else CELLS + ("WORLD_SILENT",)
        unit[name], dropped[name], inits[name] = units(targets, cells)

    stats, pv = [], []
    for j, (pop, x, ref, c) in enumerate(FAMILY, 1):
        u = unit[pop]
        ok = [i for i in range(len(inits[pop]))
              if np.isfinite(u[x][c][i]) and np.isfinite(u[ref][c][i])]
        va = np.array([u[x][c][i] for i in ok])
        vb = np.array([u[ref][c][i] for i in ok])
        delta, p = paired_sign_flip(va, vb) if ok else (float("nan"), 1.0)
        stats.append({"n": j, "population": pop, "interpreter": x, "reference": ref, "cell": c,
                      "n_units": len(ok), "delta": delta,
                      "a": float(va.mean()) if ok else float("nan"),
                      "b": float(vb.mean()) if ok else float("nan")})
        pv.append(p if np.isfinite(p) else 1.0)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
    by = {r["n"]: r for r in stats}
    positive = lambda r: r["reject"] and r["delta"] > 0
    read_out = {}
    for q, rows_q in QUESTIONS.items():
        rs = [by[n] for n in rows_q]
        if all(positive(r) for r in rs):
            read_out[q] = "SUPPORTED"
        elif any(not positive(r) and r["n_units"] < MIN_UNITS for r in rs):
            read_out[q] = "NOT TESTABLE"
        else:
            read_out[q] = "NOT SUPPORTED"
    d = by[DYN]
    read_out["Q6"] = (("better" if d["delta"] > 0 else "worse") if d["reject"]
                      else "no detectable change")

    counts = {pop: {c: int(sum(t["n_states"][c] for t in ts)) for c in ts[0]["n_states"]}
              for pop, ts in pops.items()}
    for pop in pops:
        cells = list(unit[pop][next(iter(unit[pop]))])
        print(f"\n{pop}: {len(inits[pop])} initialisations; states {counts[pop]}; "
              f"dropped {dropped[pop]}")
        print(f"{'':26s}" + "".join(f"{c:>13s}" for c in cells))
        for arm in unit[pop]:
            print(f"{arm:26s}" + "".join(f"{np.nanmean(unit[pop][arm][c]):>13.4f}"
                                         if np.isfinite(unit[pop][arm][c]).any() else f"{'-':>13s}"
                                         for c in cells))
    print(f"\n{'#':>3s} {'population':10s}{'comparison':34s}{'cell':12s}{'units':>6s}"
          f"{'delta':>9s}{'p':>10s}{'p_adj':>10s}")
    for r in stats:
        print(f"{r['n']:>3d} {r['population']:10s}{r['interpreter'] + ' vs ' + r['reference']:34s}"
              f"{r['cell']:12s}{r['n_units']:>6d}{r['delta']:>+9.4f}{r['p']:>10.5f}"
              f"{r['p_adj']:>10.5f}")
    print("\nREAD-OUT: " + "  ".join(f"{q} {v}" for q, v in read_out.items()))
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0

    out = {"provenance": {"commit": S.git("rev-parse", "HEAD"),
                          "protocol_commit": S.git("log", "-1", "--format=%H", "--", here),
                          "protocol_commit_date": S.git("log", "-1", "--format=%cI", "--", here),
                          "started_utc": started, "script": here,
                          "script_sha256": S.sha256_file(__file__),
                          "car_manifest": MANIFEST, "car_files": F,
                          "skirmish_fit": {"path": S.FIT_FILE, "sha256": sk_fit_sha,
                                           "initialisations": list(S.FIT_INITS)},
                          "section_3_2_held": {"path": S.HELD_FILE, "sha256": sk_32_sha},
                          "unit": "initialisation", "margin": MARGIN,
                          "interpreter_seeds": list(seeds), "epochs": epochs,
                          "constant_cause": {"car": maj_car, "skirmish": maj_sk},
                          "state_counts": counts, "dropped_units": dropped,
                          "torch": torch.__version__, "numpy": np.__version__,
                          "python": platform.python_version()},
           "unit_values": unit, "initialisations": inits, "comparisons": stats,
           "read_out": read_out, "per_target": pops, "wall_seconds": time.time() - t0}
    with open(OUT, "x") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
