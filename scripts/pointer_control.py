#!/usr/bin/env python
"""The first-order pointer in the paper's tables, the interpreter where the pointer is wrong,
95% intervals for the main tables, and three post-hoc analyses of the Car study.

PROTOCOL. Committed to the repository before this script is run, once.
---------------------------------------------------------------------------
Why.        On 2026-10-04 the author decided that the paper reports the first-order pointer
            as a control method again (tables 5, 6, 8, 9, section 2.6 and the discussion),
            adds it to table 7 and to section 3.7, measures the interpreter on the states
            where the pointer is wrong, and gives 95% intervals for tables 6, 7, 11 and 14.
            Some pointer numbers of the earlier paper (the |signed sum| rule on the pinned
            populations, tables 5 and 9; the rule comparison 62.6 / 73.7 / 88.7 and the
            integrated-gradient pointer 91.4 in the discussion) came from a diagnostic that
            was never committed; they are recomputed here from the pinned target models.
            Section 3.7's pointer needs the language models: scripts/pointer_sentences.py.
Pointer rules. For every cause f with input carriers C_f (follows_b.carriers_in), from the
            per-input first-order effect fo of moving each input to the all-neutral
            reference (follows_b.prepare, the signal the interpreter reads):
              meanabs    argmax_f mean_{i in C_f} |fo_i|   (follows_b's first_order_pointer)
              sumabs     argmax_f  sum_{i in C_f} |fo_i|
              signed     argmax_f |sum_{i in C_f}  fo_i|   (the paper's first-order pointer)
              ig_signed  signed, with the integrated gradient along the 16-segment path to
                         the same reference (mint.encoding.dynamics.path_signals) as fo.
            Ties go to the first cause. Scored on the margin-kept states against B's own
            cause, exactly as the interpreter.
Populations, each scored exactly as the result it extends:
  P1  section 3.1 / table 5: results/cache/skirmish-mlp-6d7fca3aed.pt (follows_b's pinned
      population), held models 12..23, unit = model, cells ALL, AGREE, DISAGREE (B against
      the world's decisive factor). Rules meanabs, sumabs, signed, ig_signed.
      CHECK 1: meanabs equals results/follows_b.json first_order_pointer per model to 1e-9.
  P2  table 9, medical targets: results/cache/clinic-mlp-6326cb5796.pt, held 12..23, unit =
      model, cells ALL, AGREE, DISAGREE. Rules meanabs, signed. CHECK 2: meanabs equals
      results/transfer_b.json first_order_pointer "skirmish->clinic" per model to 1e-9.
      The tactical targets of table 9 are P1's.
  P3  sections 3.2 / tables 6 and 7: seeds12_all's populations (fit 8..19 x 3, held 40..51
      x 3), unit = initialisation (mean over its three targets with states in the cell),
      cells ALL, AGREE, DISAGREE, B_SPECIFIC (typical_control.typical over the 36 fit
      targets). Rule signed. CHECK 3: signed equals results/seeds12_all.json per_target
      check_first_order_signed in ALL, AGREE, DISAGREE to 1e-9.
  P4  section 3.8 / table 14: car_study's Car populations, cells and units exactly as
      car_study. Rule signed. CHECK 4: signed equals results/car_study.json per_target
      check_first_order_signed in every stored cell to 1e-9.
Interpreter where the pointer is wrong. The interpreters of P1 (follows_b: fit models 0..11),
            P3 (seeds12_all) and P4 (car_study's `car`) are refitted with their recipes
            unchanged (seeds 0, 1, 2; 4 threads per worker). CONTROL: their per-target
            accuracies must equal results/follows_b.json, results/seeds12_all.json and
            results/car_study.json in every stored cell to 1e-9; otherwise the run stops
            and nothing is written. Reported per cell: the interpreter's accuracy on states
            where the signed pointer is wrong, the pointer's accuracy on states where the
            interpreter is wrong, and the number of states in each set; per unit and as the
            mean over units. The interpreter's correctness at a state is the mean over its
            three seeds; a state counts as one where the interpreter is wrong when that mean
            is below 0.5 (at most one seed names B's cause).
Intervals.  95% percentile bootstrap over the independent units (10 000 resamples; every
            interval draws from its own numpy default_rng(0)) of the mean of every arm in
            every cell and of the mean paired difference of every declared comparison, for
            results/seeds12_all.json (table 6), results/typical_control.json (table 7),
            results/dynamics_deep.json (table 11) and results/car_study.json (table 14), and
            for the new pointer rows of P3 and P4.
Post hoc, Car. Requested by an audit after the Car run; they describe, they do not test.
            (a) The median gap between B's first and second cause (follows_b's margin) on
            B_SPECIFIC states and on the other kept states, for Car and for the reverse-
            transfer skirmish targets 60..71 (typical over the 36 skirmish fit targets).
            (b) How Car's B_SPECIFIC splits into states where the world names a single
            cause and states where it names none, with the interpreter's and the constant's
            pooled accuracy in each part.
            (c) An architecture-matched typical oracle (the cause most of the 12 Car fit
            targets of the same architecture rely on, ties by mean |effect|) on Car's
            DISAGREE states, pooled per architecture, against the interpreter on the same
            states.
Everything above is reported; nothing enters a read-out; there is no test.
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/pointer_control.py
            It refuses to run unless this file equals the committed blob and no tracked
            file is modified, and it refuses to overwrite results/pointer_control.json.
            `--smoke`: two held targets per population, small fit sets, 2 epochs, one seed,
            no CHECK and no CONTROL, 200 resamples, writes nothing.
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

import car_study as CS
import dynamics_deep as DD
import seeds12_all as S
import typical_control as TC
from follows_b import (MARGIN, carriers_in, load_population, named, prepare, tensors,
                       train_pointer)
from mint.domains import get_domain
from mint.encoding.dynamics import path_signals

OUT = "results/pointer_control.json"
CACHE = "results/cache/pointer_control/packs.pt"
WORKERS, THREADS = 5, 4
RULES = ("meanabs", "sumabs", "signed")
_G: dict = {}


def rule(fo: np.ndarray, car: list[list[int]], kind: str) -> np.ndarray:
    if kind == "meanabs":
        s = [np.abs(fo[:, c]).mean(1) for c in car]
    elif kind == "sumabs":
        s = [np.abs(fo[:, c]).sum(1) for c in car]
    else:
        s = [np.abs(fo[:, c].sum(1)) for c in car]
    return np.stack(s, 1).argmax(1)


def acc(ok: np.ndarray, m: np.ndarray) -> float:
    return float(np.asarray(ok, float)[m].mean()) if m.any() else float("nan")


def boot(v, n: int) -> list[float]:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_diff(a, b, n: int) -> list[float]:
    d = [x - y for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    return boot(d, n)


# ---------------------------------------------------------------- refitting in workers
def _init(cache: str, threads: int) -> None:
    torch.set_num_threads(threads)
    _G["packs"] = torch.load(cache, weights_only=False)


def job(args):
    pop, seed, epochs = args
    P = _G["packs"][pop]
    spec = get_domain(P["domain"]).spec
    t0 = time.time()
    m = train_pointer(tensors(P["fit"], P["u_max"], spec, MARGIN), spec, seed=seed, epochs=epochs)
    preds = [named(m, p, spec, P["u_max"], MARGIN) for p in P["held"]]
    print(f"  {pop} seed {seed} refitted ({time.time() - t0:.0f}s)", flush=True)
    return (pop, seed), preds


def slim(p: dict) -> dict:
    return {k: p[k] for k in ("unit_x", "pos_x", "global_x", "y", "world", "margin")}


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
    seeds, epochs, n_boot = ((0,), 2, 200) if a.smoke else (S.SEEDS, S.EPOCHS, 10_000)
    nh = 2 if a.smoke else None
    out: dict = {"checks": {}, "controls": {}}

    # ------------------------------------------------------------ P1: section 3.1
    sk = get_domain("skirmish")
    b1, p1_path, p1_sha = load_population("skirmish", 24)
    k1 = [prepare(b, sk) for b in b1]
    p1_fit, p1_held, b1_held = k1[:12], k1[12:24][:nh], b1[12:24][:nh]
    car_sk = carriers_in(sk.spec, k1[0]["pos_x"].shape[1])
    u1 = max(p["unit_x"].shape[1] for p in k1)
    p1 = {r: [] for r in RULES + ("ig_signed",)}
    p1_states = []
    for b, p in zip(b1_held, p1_held):
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], p["world"][k] == p["y"][k]
        ig = path_signals(b.trained.model, sk, b.focal_raw, b.gt.probe.act_std,
                          b.gt.probe.logit_std, 16)["ig"][k]
        preds = {r: rule(p["fo"][k], car_sk, r) for r in RULES} | {"ig_signed": rule(ig, car_sk, "signed")}
        masks = {"ALL": np.ones(len(y), bool), "AGREE": agree, "DISAGREE": ~agree}
        for r, pr in preds.items():
            p1[r].append({c: acc(pr == y, m) for c, m in masks.items()})
        p1_states.append({"y": y, "masks": masks, "signed": preds["signed"]})
    if not a.smoke:
        ref = {r["cell"]: [m["accuracy"] for m in r["per_model"]]
               for r in json.load(open("results/follows_b.json"))["rows"]
               if r["method"] == "first_order_pointer"}
        for c in ("ALL", "AGREE", "DISAGREE"):
            got = [t[c] for t in p1["meanabs"] if np.isfinite(t[c])]
            if not np.allclose(got, ref[c], rtol=0, atol=1e-9):
                print(f"CHECK 1 FAILED in {c}. Stopping.")
                return 4
        out["checks"]["1"] = "passed"
    out["P1"] = {r: {"per_model": v, "mean": {c: float(np.nanmean([t[c] for t in v]))
                                               for c in ("ALL", "AGREE", "DISAGREE")}}
                 for r, v in p1.items()}
    print(f"P1 pointer rules ({time.time() - t0:.0f}s): " + "; ".join(
        f"{r} {out['P1'][r]['mean']['ALL']:.4f}/{out['P1'][r]['mean']['DISAGREE']:.4f}" for r in p1),
        flush=True)

    # ------------------------------------------------------------ P2: medical targets
    cl = get_domain("clinic")
    b2, p2_path, p2_sha = load_population("clinic", 24)
    k2 = [prepare(b, cl) for b in b2[12:24][:nh]]
    car_cl = carriers_in(cl.spec, k2[0]["pos_x"].shape[1])
    p2 = {r: [] for r in ("meanabs", "signed")}
    for p in k2:
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], p["world"][k] == p["y"][k]
        masks = {"ALL": np.ones(len(y), bool), "AGREE": agree, "DISAGREE": ~agree}
        for r in p2:
            p2[r].append({c: acc(rule(p["fo"][k], car_cl, r) == y, m) for c, m in masks.items()})
    if not a.smoke:
        rows = json.load(open("results/transfer_b.json"))["rows"]
        for c in ("ALL", "AGREE", "DISAGREE"):
            ref = [m["accuracy"] for r in rows if r["method"] == "first_order_pointer"
                   and r["cell"] == f"skirmish->clinic|{c}" for m in r["per_model"]]
            got = [t[c] for t in p2["meanabs"] if np.isfinite(t[c])]
            if not np.allclose(got, ref, rtol=0, atol=1e-9):
                print(f"CHECK 2 FAILED in {c}. Stopping.")
                return 4
        out["checks"]["2"] = "passed"
    out["P2"] = {r: {"per_model": v, "mean": {c: float(np.nanmean([t[c] for t in v]))
                                               for c in ("ALL", "AGREE", "DISAGREE")}}
                 for r, v in p2.items()}
    print(f"P2 medical targets: signed {out['P2']['signed']['mean']['ALL']:.4f}/"
          f"{out['P2']['signed']['mean']['DISAGREE']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ P3: section 3.2
    fit_man, held_man = json.load(open(S.FIT_MANIFEST)), json.load(open(S.HELD_MANIFEST))
    fit_b, fit_sha = S.load_checked(S.FIT_FILE, fit_man["files"]["new_models"]["sha256"])
    held_b, held_sha = S.load_checked(S.HELD_FILE, held_man["file"]["sha256"])
    fit_idx = S.indices(fit_b, S.FIT_INITS, S.FIT_BASE)
    held_idx = S.indices(held_b, S.HELD_INITS, S.HELD_BASE)
    if a.smoke:
        fit_idx, held_idx = fit_idx[:6], held_idx[:2]
    k3_fit = [prepare(fit_b[i], sk) for i in fit_idx]
    k3_held = [prepare(held_b[i], sk) for i in held_idx]
    u3 = max(p["unit_x"].shape[1] for p in k3_fit + k3_held)
    fit3 = [fit_b[i] for i in fit_idx]
    p3_targets = []
    for i, p in zip(held_idx, k3_held):
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], p["world"][k] == p["y"][k]
        typ = TC.typical(fit3, sk, held_b[i].focal_raw[k])
        sg = rule(p["fo"][k], car_sk, "signed")
        masks = {"ALL": np.ones(len(y), bool), "AGREE": agree, "DISAGREE": ~agree,
                 "B_SPECIFIC": typ != y}
        cfg = held_b[i].trained.cfg
        p3_targets.append({"initialisation": cfg.seed, "strength": float(cfg.spurious_strength),
                           "y": y, "masks": masks, "signed": sg,
                           "pointer": {c: acc(sg == y, m) for c, m in masks.items()}})
    if not a.smoke:
        ref = json.load(open(S.OUT))["per_target"]
        for t, r in zip(p3_targets, ref):
            for c in ("ALL", "AGREE", "DISAGREE"):
                if not np.isclose(t["pointer"][c], r["accuracy"]["check_first_order_signed"][c],
                                  rtol=0, atol=1e-9, equal_nan=True):
                    print(f"CHECK 3 FAILED at {t['initialisation']}/{t['strength']} {c}. Stopping.")
                    return 4
        out["checks"]["3"] = "passed"
    print(f"P3 pointer measured on {len(p3_targets)} targets ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ P4: Car
    car = get_domain("car")
    man = json.load(open(CS.MANIFEST))
    F = man["files"]
    car_fit = CS.load_pt(F["fit"]["path"], F["fit"]["sha256"])
    car_held = CS.load_pt(F["held"]["path"], F["held"]["sha256"])
    fit4_b, held4_b = car_fit["bundles"], car_held["bundles"]
    fit4_rec, held4_rec = car_fit["records"], car_held["records"]
    if a.smoke:
        fit4_b, fit4_rec, held4_b, held4_rec = fit4_b[:6], fit4_rec[:6], held4_b[:2], held4_rec[:2]
    k4_fit = [prepare(b, car) for b in fit4_b]
    k4_held = [prepare(b, car) for b in held4_b]
    u4 = max(p["unit_x"].shape[1] for p in k4_fit + k4_held)
    if not a.smoke and u4 != 60:
        raise RuntimeError(f"Car padding {u4} differs from car_study's 60")
    car_car = carriers_in(car.spec, car.spec.n_inputs)
    typ_car = TC.typical(fit4_b, car, car.states)
    cs = json.load(open("results/car_study.json"))
    maj_car = int(cs["provenance"]["constant_cause"]["car"])
    p4_targets = []
    for b, p, rec in zip(held4_b, k4_held, held4_rec):
        k = p["margin"] > MARGIN
        rows = np.flatnonzero(k)
        y, world = p["y"][k], p["world"][k]
        named_w = (b.gt.decisive_margin > 0)[k]
        sg = rule(p["fo"][k], car_car, "signed")
        masks = {"ALL": np.ones(len(y), bool), "AGREE": named_w & (world == y),
                 "DISAGREE": named_w & (world != y), "B_SPECIFIC": typ_car[rows] != y,
                 "WORLD_SILENT": ~named_w, "UNSEEN": ~np.isin(rows, rec["train_rows"])}
        p4_targets.append({"initialisation": rec["initialisation"], "arch": rec["arch"],
                           "y": y, "rows": rows, "named": named_w, "masks": masks, "signed": sg,
                           "margin": p["margin"][k],
                           "pointer": {c: acc(sg == y, m) for c, m in masks.items()}})
    if not a.smoke:
        for t, r in zip(p4_targets, cs["per_target"]["car"]):
            for c, v in t["pointer"].items():
                if not np.isclose(v, r["accuracy"]["check_first_order_signed"][c], rtol=0,
                                  atol=1e-9, equal_nan=True):
                    print(f"CHECK 4 FAILED at {t['initialisation']}/{t['arch']} {c}. Stopping.")
                    return 4
        out["checks"]["4"] = "passed"
    print(f"P4 pointer measured on {len(p4_targets)} targets ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ refit the interpreters
    packs = {"P1": {"domain": "skirmish", "fit": [slim(p) for p in p1_fit],
                    "held": [slim(p) for p in p1_held], "u_max": u1},
             "P3": {"domain": "skirmish", "fit": [slim(p) for p in k3_fit],
                    "held": [slim(p) for p in k3_held], "u_max": u3},
             "P4": {"domain": "car", "fit": [slim(p) for p in k4_fit],
                    "held": [slim(p) for p in k4_held], "u_max": u4}}
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(packs, CACHE)
    jobs = [(pop, s, epochs) for pop in ("P3", "P4", "P1") for s in seeds]
    with mp.get_context("spawn").Pool(min(WORKERS, len(jobs)), initializer=_init,
                                      initargs=(CACHE, THREADS)) as pool:
        preds = dict(pool.map(job, jobs))
    print(f"interpreters refitted ({time.time() - t0:.0f}s)", flush=True)
    ok_of = lambda pop, t, y: np.mean([preds[(pop, s)][t] == y for s in seeds], 0)

    # CONTROL
    p1_ok = [ok_of("P1", t, st["y"]) for t, st in enumerate(p1_states)]
    p3_ok = [ok_of("P3", t, tg["y"]) for t, tg in enumerate(p3_targets)]
    p4_ok = [ok_of("P4", t, tg["y"]) for t, tg in enumerate(p4_targets)]
    if not a.smoke:
        ref = {r["cell"]: [m["accuracy"] for m in r["per_model"]]
               for r in json.load(open("results/follows_b.json"))["rows"] if r["method"] == "interpreter"}
        for c in ("ALL", "AGREE", "DISAGREE"):
            got = [acc(ok, st["masks"][c]) for ok, st in zip(p1_ok, p1_states)]
            got = [g for g in got if np.isfinite(g)]
            if not np.allclose(got, ref[c], rtol=0, atol=1e-9):
                print(f"CONTROL FAILED: P1 {c}. Stopping before anything is written.")
                return 5
        ref = json.load(open(S.OUT))["per_target"]
        for ok, tg, r in zip(p3_ok, p3_targets, ref):
            for c in ("ALL", "AGREE", "DISAGREE"):
                if not np.isclose(acc(ok, tg["masks"][c]), r["accuracy"]["interpreter"][c],
                                  rtol=0, atol=1e-9, equal_nan=True):
                    print(f"CONTROL FAILED: P3 {tg['initialisation']} {c}. Stopping.")
                    return 5
        for ok, tg, r in zip(p4_ok, p4_targets, cs["per_target"]["car"]):
            for c in tg["masks"]:
                if not np.isclose(acc(ok, tg["masks"][c]), r["accuracy"]["car"][c],
                                  rtol=0, atol=1e-9, equal_nan=True):
                    print(f"CONTROL FAILED: P4 {tg['initialisation']}/{tg['arch']} {c}. Stopping.")
                    return 5
        out["controls"] = {"P1": "passed", "P3": "passed", "P4": "passed"}
        print("control passed: the refitted interpreters reproduce follows_b, seeds12_all and "
              "car_study on every target", flush=True)

    # ------------------------------------------------------------ where the pointer is wrong
    def crossing(ok, y, signed, masks):
        pw = signed != y
        iw = np.asarray(ok) < 0.5
        return {c: {"interp_where_pointer_wrong": acc(ok, m & pw),
                    "pointer_where_interp_wrong": acc(signed == y, m & iw),
                    "n_pointer_wrong": int((m & pw).sum()), "n_interp_wrong": int((m & iw).sum()),
                    "n": int(m.sum())} for c, m in masks.items()}

    def by_unit(targets, rows_of, cells):
        inits = sorted({t["initialisation"] for t in targets})
        res = {}
        for c in cells:
            per = []
            for s in inits:
                block = [r for t, r in zip(targets, rows_of) if t["initialisation"] == s and r[c]["n"]]
                per.append({k: (float(np.nanmean([r[c][k] for r in block])) if block else float("nan"))
                            for k in ("interp_where_pointer_wrong", "pointer_where_interp_wrong")})
            res[c] = {"per_unit": per,
                      "mean": {k: float(np.nanmean([u[k] for u in per]))
                               for k in ("interp_where_pointer_wrong", "pointer_where_interp_wrong")},
                      "states": {k: int(sum(r[c][k] for r in rows_of))
                                 for k in ("n_pointer_wrong", "n_interp_wrong", "n")}}
        return res

    x1 = [crossing(ok, st["y"], st["signed"], st["masks"]) for ok, st in zip(p1_ok, p1_states)]
    out["crossing_P1"] = {c: {"mean": {k: float(np.nanmean([r[c][k] for r in x1]))
                                      for k in ("interp_where_pointer_wrong", "pointer_where_interp_wrong")},
                              "states": {k: int(sum(r[c][k] for r in x1))
                                         for k in ("n_pointer_wrong", "n_interp_wrong", "n")}}
                          for c in ("ALL", "AGREE", "DISAGREE")}
    x3 = [crossing(ok, tg["y"], tg["signed"], tg["masks"]) for ok, tg in zip(p3_ok, p3_targets)]
    out["crossing_P3"] = by_unit(p3_targets, x3, ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC"))
    x4 = [crossing(ok, tg["y"], tg["signed"], tg["masks"]) for ok, tg in zip(p4_ok, p4_targets)]
    out["crossing_P4"] = by_unit(p4_targets, x4, ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC"))

    # ------------------------------------------------------------ pointer rows of P3, P4 by unit
    def pointer_units(targets, cells):
        inits = sorted({t["initialisation"] for t in targets})
        res = {}
        for c in cells:
            per = []
            for s in inits:
                vals = [t["pointer"][c] for t in targets if t["initialisation"] == s
                        and np.isfinite(t["pointer"][c])]
                per.append(float(np.mean(vals)) if vals else float("nan"))
            res[c] = {"per_unit": per, "mean": float(np.nanmean(per)), "ci95": boot(per, n_boot)}
        return res

    out["P3_pointer"] = pointer_units(p3_targets, ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC"))
    out["P4_pointer"] = pointer_units(p4_targets, ("ALL", "AGREE", "DISAGREE", "B_SPECIFIC",
                                                   "WORLD_SILENT", "UNSEEN"))

    # ------------------------------------------------------------ intervals for tables 6, 7, 11, 14
    iv: dict = {}
    s12 = json.load(open(S.OUT))
    units12 = {(r["method"], r["cell"]): {m["initialisation"]: m["accuracy"] for m in r["per_model"]}
               for r in s12["rows"]}
    inits = list(S.HELD_INITS)
    vec = lambda arm, c: [units12.get((arm, c), {}).get(s, float("nan")) for s in inits]
    iv["table6"] = {"arms": {f"{arm}|{c}": boot(vec(arm, c), n_boot) for (arm, c) in units12},
                    "comparisons": {f"interpreter-{ref}|{c}": boot_diff(vec("interpreter", c), vec(ref, c), n_boot)
                                    for ref, c in S.FAMILY}}
    tc = json.load(open(TC.OUT))["unit_values"]
    iv["table7"] = {"arms": {f"{arm}|{c}": boot(v, n_boot) for arm, cs_ in tc.items() for c, v in cs_.items()},
                    "comparisons": {f"interpreter-{ref}|{c}": boot_diff(tc["interpreter"][c], tc[ref][c], n_boot)
                                    for ref, c in TC.FAMILY}}
    dd = json.load(open(DD.OUT))["unit_values"]
    iv["table11"] = {"arms": {f"{arm}|{c}": boot(v, n_boot) for arm, cs_ in dd.items() for c, v in cs_.items()},
                     "comparisons": {f"{x}-{ref}|{c}": boot_diff(dd[x][c], dd[ref][c], n_boot)
                                     for x, ref, c in DD.FAMILY}}
    cu = cs["unit_values"]
    iv["table14"] = {"arms": {f"{pop}|{arm}|{c}": boot(v, n_boot) for pop, arms in cu.items()
                              for arm, cs_ in arms.items() for c, v in cs_.items()},
                     "comparisons": {f"{n}:{pop}|{x}-{ref}|{c}": boot_diff(cu[pop][x][c], cu[pop][ref][c], n_boot)
                                     for n, (pop, x, ref, c) in enumerate(CS.FAMILY, 1)}}
    out["intervals"] = iv

    # ------------------------------------------------------------ post hoc, Car
    ph: dict = {}
    gap_bs = np.concatenate([t["margin"][t["masks"]["B_SPECIFIC"]] for t in p4_targets])
    gap_ot = np.concatenate([t["margin"][~t["masks"]["B_SPECIFIC"]] for t in p4_targets])
    ph["a_gap_car"] = {"b_specific_median": float(np.median(gap_bs)), "other_median": float(np.median(gap_ot)),
                       "n": [int(len(gap_bs)), int(len(gap_ot))]}
    sk_held_b = CS.load_pt(F["skirmish_held"]["path"], F["skirmish_held"]["sha256"])[:nh]
    sk_fit_b = [fit_b[i] for i in S.indices(fit_b, S.FIT_INITS, S.FIT_BASE)][: (6 if a.smoke else None)]
    g_bs, g_ot = [], []
    for b in sk_held_b:
        p = prepare(b, sk)
        k = p["margin"] > MARGIN
        typ = TC.typical(sk_fit_b, sk, b.focal_raw[k])
        bs = typ != p["y"][k]
        g_bs.append(p["margin"][k][bs]); g_ot.append(p["margin"][k][~bs])
    g_bs, g_ot = np.concatenate(g_bs), np.concatenate(g_ot)
    ph["a_gap_skirmish_60_71"] = {"b_specific_median": float(np.median(g_bs)) if len(g_bs) else float("nan"),
                                  "other_median": float(np.median(g_ot)), "n": [int(len(g_bs)), int(len(g_ot))]}
    parts = {"named": [], "silent": []}
    for ok, t in zip(p4_ok, p4_targets):
        bs = t["masks"]["B_SPECIFIC"]
        for name_, m in (("named", bs & t["named"]), ("silent", bs & ~t["named"])):
            parts[name_].append((np.asarray(ok)[m], (t["y"][m] == maj_car).astype(float)))
    ph["b_split"] = {k: {"n": int(sum(len(o) for o, _ in v)),
                         "interpreter": float(np.concatenate([o for o, _ in v]).mean()) if v else float("nan"),
                         "constant": float(np.concatenate([c for _, c in v]).mean()) if v else float("nan")}
                     for k, v in parts.items()}
    ph["b_split"]["disagree_inside_b_specific"] = bool(all(
        not (t["masks"]["DISAGREE"] & ~t["masks"]["B_SPECIFIC"]).any() for t in p4_targets))
    arch_res = {}
    for arch in sorted({r["arch"] for r in held4_rec}):
        fit_a = [b for b, r in zip(fit4_b, fit4_rec) if r["arch"] == arch]
        typ_a = TC.typical(fit_a, car, car.states)
        o_i, o_a = [], []
        for ok, t in zip(p4_ok, p4_targets):
            if t["arch"] != arch:
                continue
            m = t["masks"]["DISAGREE"]
            o_i.append(np.asarray(ok)[m]); o_a.append((typ_a[t["rows"]][m] == t["y"][m]).astype(float))
        o_i, o_a = np.concatenate(o_i), np.concatenate(o_a)
        arch_res[str(arch)] = {"n": int(len(o_i)),
                               "interpreter": float(o_i.mean()) if len(o_i) else float("nan"),
                               "arch_typical_oracle": float(o_a.mean()) if len(o_a) else float("nan")}
    ph["c_arch_matched_typical_on_disagree"] = arch_res
    out["post_hoc_car"] = ph

    print(json.dumps({"P1_ALL": {r: round(v["mean"]["ALL"], 4) for r, v in out["P1"].items()},
                      "P2_signed": out["P2"]["signed"]["mean"],
                      "P3_pointer": {c: round(v["mean"], 4) for c, v in out["P3_pointer"].items()},
                      "P4_pointer": {c: round(v["mean"], 4) for c, v in out["P4_pointer"].items()},
                      "crossing_P3": {c: v["mean"] for c, v in out["crossing_P3"].items()},
                      "post_hoc_car": ph}, indent=1, default=float))
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0
    out["provenance"] = {"commit": S.git("rev-parse", "HEAD"),
                         "protocol_commit": S.git("log", "-1", "--format=%H", "--", here),
                         "protocol_commit_date": S.git("log", "-1", "--format=%cI", "--", here),
                         "started_utc": started, "script": here, "script_sha256": S.sha256_file(__file__),
                         "populations": {"P1": [p1_path, p1_sha], "P2": [p2_path, p2_sha],
                                         "P3": {"fit": [S.FIT_FILE, fit_sha], "held": [S.HELD_FILE, held_sha]},
                                         "P4": F},
                         "seeds": list(seeds), "epochs": epochs, "margin": MARGIN, "bootstrap": n_boot,
                         "torch": torch.__version__, "numpy": np.__version__,
                         "python": platform.python_version()}
    out["wall_seconds"] = time.time() - t0
    with open(OUT, "x") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
