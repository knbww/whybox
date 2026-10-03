#!/usr/bin/env python
"""Does the interpreter name the cause better when it reads dynamic signals of B?

Protocol: `docs/PREREG_SIGNALS.md`. The full run refuses to start unless the
protocol and the code it depends on are committed and unmodified.

Why this run exists. `results/volume_study.json`: eight times more training
volume, of either kind, moved accuracy by +0.007, and the interpreter sat on the
same plateau (train ~0.77, validation ~0.73) at every rung. The limit is what the
signals carry, not how many examples it sees. Everything it reads today is a
snapshot at the state; for a ReLU target that snapshot is blind to what happens
between the state and the reference. `mint.encoding.dynamics` measures that path.

Four arms, one interpreter. Only the inputs differ: architecture, hyperparameters,
training states (the twelve pinned fitted models, 160 states each), stopping rule
and held-out targets are identical.

    current      what the confirmed interpreter reads
    +snapshot    + each unit's activation and gradient at the state, separately
                   (more of the same moment -- the control for "just more inputs")
    +dynamics    + path signals: integrated gradients and gradient variability per
                   input position, conductance and activation response per unit,
                   the total change of B's output along the path
    +both        + snapshot and dynamics

Every new per-unit and per-position quantity is expanded exactly as the existing
ones are (signed and absolute value over the state's maximum, tied rank, share,
sign), so no arm gets a different kind of normalisation.

    python -m scripts.signals_study            # full run, protocol must be committed
    python -m scripts.signals_study --smoke    # 2 models, 2 epochs, end-to-end
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mint.domains import get_domain
from mint.encoding.dynamics import path_signals
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from scripts.follows_b import MARGIN, load_population, pos_features, prepare, recover_name, tensors
from scripts.volume_study import _git, batched_names, committed_and_clean, fit, load_manifest

DOMAIN = "skirmish"
PREREG = "docs/PREREG_SIGNALS.md"
ARMS = ("current", "+snapshot", "+dynamics", "+both")
FULL = dict(manifest="results/volume_manifest.json", cache="results/cache/signals/packs.pt",
            out="results/signals_study.json", n_fit=12, n_held=12, n_val=12, steps=16,
            seeds=(0, 1, 2), max_epochs=200, patience=20, workers=6, threads=3)
SMOKE = dict(manifest="results/cache/volume_smoke/manifest.json",
             cache="results/cache/signals_smoke/packs.pt",
             out="results/cache/signals_smoke/signals_study_smoke.json", n_fit=2, n_held=2,
             n_val=2, steps=4, seeds=(0,), max_epochs=2, patience=1, workers=2, threads=2,
             held_from_fit=True)
FAMILY = [(arm, sub) for arm in ARMS[1:] for sub in ("ALL", "DISAGREE")]
DIAGNOSTIC = [("+dynamics", "+snapshot"), ("+both", "+dynamics")]
SUBSETS = ("ALL", "AGREE", "DISAGREE")
CLIP = 10.0
COMMITTED = (PREREG, "scripts/signals_study.py", "src/mint/encoding/dynamics.py",
             "tests/test_dynamics.py", "scripts/volume_study.py", "scripts/follows_b.py",
             "results/volume_manifest.json")

_G: dict = {}


def signal_pack(bundle, d, steps: int) -> dict:
    """Everything any arm may read about one target model, plus what scoring needs."""
    base = prepare(bundle, d)
    gt, probe = bundle.gt, bundle.gt.probe
    s = path_signals(bundle.trained.model, d, bundle.focal_raw, probe.act_std, probe.logit_std, steps)
    z = np.clip(np.nan_to_num((gt.acts - probe.act_mean) / (probe.act_std + 1e-6)), -CLIP, CLIP)
    gn = np.clip(np.nan_to_num(gt.grads / (probe.grad_absmean + 1e-9)), -CLIP, CLIP)
    return base | {
        "snap_u": np.concatenate([pos_features(z), pos_features(gn)], -1),
        "dyn_u": np.concatenate([pos_features(s["cond"]),
                                 pos_features(np.clip(s["dact"], -CLIP, CLIP))], -1),
        "dyn_p": np.concatenate([pos_features(s["ig"]), pos_features(s["gvar"])], -1),
        "dyn_g": np.stack([s["total"], np.log1p(np.abs(s["ig"]).sum(1))], 1).astype(np.float32),
        "ig": s["ig"],
    }


def arm_view(p: dict, arm: str) -> dict:
    ux, px, gx = [p["unit_x"]], [p["pos_x"]], [p["global_x"]]
    if arm in ("+snapshot", "+both"):
        ux.append(p["snap_u"])
    if arm in ("+dynamics", "+both"):
        ux.append(p["dyn_u"]); px.append(p["dyn_p"]); gx.append(p["dyn_g"])
    keep = ("y", "world", "margin")
    return {k: p[k] for k in keep} | {"unit_x": np.concatenate(ux, -1),
                                      "pos_x": np.concatenate(px, -1),
                                      "global_x": np.concatenate(gx, -1)}


def build(c: dict) -> tuple[dict, dict]:
    d = get_domain(DOMAIN)
    t0 = time.time()
    pinned, path, sha = load_population(DOMAIN, 24)
    man = load_manifest(c["manifest"])
    val_b = torch.load(man["files"]["val_models"]["path"], weights_only=False)[:c["n_val"]]
    fit_b = pinned[:c["n_fit"]]
    if c.get("held_from_fit"):
        # smoke only: stand-ins from the fitted pool, so a smoke test never prints a
        # number about a held-out model
        held_b = pinned[c["n_fit"]:c["n_fit"] + c["n_held"]]
    else:
        held_b = pinned[12:12 + c["n_held"]]
        assert {b.trained.cfg.seed for b in held_b} <= {4, 5, 6, 7}
    assert not {b.trained.cfg.seed for b in fit_b + val_b} & {4, 5, 6, 7}
    packs = {"fit": [signal_pack(b, d, c["steps"]) for b in fit_b],
             "val": [signal_pack(b, d, c["steps"]) for b in val_b],
             "held": [signal_pack(b, d, c["steps"]) for b in held_b]}
    Path(c["cache"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(packs, c["cache"])
    h = hashlib.sha256(Path(c["cache"]).read_bytes()).hexdigest()
    print(f"signals measured for {sum(map(len, packs.values()))} models ({time.time() - t0:.0f}s)",
          flush=True)
    return packs, {"held_out_cache": {"path": path, "sha256": sha},
                   "val_models": man["files"]["val_models"], "signal_packs": {"path": c["cache"], "sha256": h}}


def _worker_init(c: dict) -> None:
    torch.set_num_threads(c["threads"])
    _G["packs"] = torch.load(c["cache"], weights_only=False)
    _G["spec"] = get_domain(DOMAIN).spec


def job(args):
    arm, seed, c = args
    spec, P = _G["spec"], _G["packs"]
    views = {k: [arm_view(p, arm) for p in P[k]] for k in ("fit", "val", "held")}
    u_max = max(v["unit_x"].shape[1] for vs in views.values() for v in vs)
    tr = tensors(views["fit"], u_max, spec, MARGIN)
    va = tensors(views["val"], u_max, spec, MARGIN)
    t0 = time.time()
    m, log = fit(tr, va, spec, seed, c["max_epochs"], c["patience"])
    preds = [batched_names(m, tensors([v], u_max, spec, MARGIN), spec) for v in views["held"]]
    log |= {"arm": arm, "seed": seed, "d_unit": int(tr[0].shape[-1]), "d_pos": int(tr[2].shape[-1]),
            "d_global": int(tr[3].shape[-1]), "n_train_states": int(len(tr[5])),
            "wall_seconds": time.time() - t0}
    print(f"  done {arm:10s} seed {seed}: epoch {log['best_epoch']}/{log['epochs_run']} "
          f"({log['wall_seconds']:.0f}s)", flush=True)
    return (arm, seed), preds, log


def per_model(held: list[dict], preds_by_seed: list[list[np.ndarray]]) -> dict[str, list[float]]:
    out = {s: [] for s in SUBSETS}
    for j, p in enumerate(held):
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], p["world"][k] == p["y"][k]
        acc = np.mean([sp[j] == y for sp in preds_by_seed], axis=0)
        for s, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree), ("DISAGREE", ~agree)):
            out[s].append(float(acc[msk].mean()) if msk.sum() else float("nan"))
    return out


def floors(packs: dict, spec) -> dict[str, list[list[np.ndarray]]]:
    """Parameter-free arms. A sanity floor, outside the family, never the claim."""
    ys = np.concatenate([p["y"][p["margin"] > MARGIN] for p in packs["fit"]])
    maj = int(np.bincount(ys, minlength=spec.n_factors).argmax())
    rng = np.random.default_rng(0)
    out = {nm: [] for nm in ("world_oracle", "first_order_pointer", "ig_pointer", "constant", "random")}
    for p in packs["held"]:
        k = p["margin"] > MARGIN
        out["world_oracle"].append(p["world"][k])
        out["first_order_pointer"].append(recover_name(np.abs(p["fo"][k]), spec))
        out["ig_pointer"].append(recover_name(np.abs(p["ig"][k]), spec))
        out["constant"].append(np.full(int(k.sum()), maj))
        out["random"].append(rng.integers(0, spec.n_factors, int(k.sum())))
    return {nm: [v] for nm, v in out.items()}


def main(smoke: bool = False) -> int:
    c = SMOKE if smoke else FULL
    torch.set_num_threads(c["threads"])
    if not smoke:
        for path in COMMITTED:
            if not committed_and_clean(path):
                print(f"refusing to run: {path} is not committed, or has uncommitted changes")
                return 2
    t0 = time.time()
    packs, sources = build(c)
    spec = get_domain(DOMAIN).spec

    jobs = [(arm, s, c) for arm in ARMS for s in c["seeds"]]
    with mp.get_context("spawn").Pool(c["workers"], initializer=_worker_init, initargs=(c,)) as pool:
        done = pool.map(job, jobs, chunksize=1)
    preds = {key: p for key, p, _ in done}
    logs = [log for _, _, log in done]

    curve = {arm: per_model(packs["held"], [preds[(arm, s)] for s in c["seeds"]]) for arm in ARMS}
    curve |= {nm: per_model(packs["held"], v) for nm, v in floors(packs, spec).items()}
    rows = []
    for nm, pm in curve.items():
        for sub in SUBSETS:
            v = [x for x in pm[sub] if np.isfinite(x)]
            rows.append({"method": nm, "role": "arm" if nm in ARMS else "floor", "cell": sub,
                         "n_models": len(v), "per_model": [{"accuracy": x} for x in pm[sub]],
                         "accuracy": float(np.mean(v)) if v else float("nan")})

    stats, pv = [], []
    for arm, sub in FAMILY:
        a, b = np.array(curve[arm][sub]), np.array(curve["current"][sub])
        delta, p = paired_sign_flip(a, b)
        stats.append({"role": "primary", "method": arm, "reference": "current", "cell": sub,
                      "n_models": int(np.isfinite(a - b).sum()), "delta": delta,
                      "a": float(np.nanmean(a)), "b": float(np.nanmean(b))})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
        r["reading"] = ("better" if r["reject"] and r["delta"] > 0 else
                        "worse" if r["reject"] and r["delta"] < 0 else "no detectable change")
    diagnostic = []
    for arm, ref in DIAGNOSTIC:
        for sub in ("ALL", "DISAGREE"):
            delta, p = paired_sign_flip(np.array(curve[arm][sub]), np.array(curve[ref][sub]))
            diagnostic.append({"role": "diagnostic", "method": arm, "reference": ref, "cell": sub,
                               "delta": delta, "p_unadjusted": p})

    print(f"\n{'arm':22s}{'ALL':>9s}{'AGREE':>9s}{'DISAGREE':>10s}")
    for r_ in ARMS + ("ig_pointer", "first_order_pointer", "constant", "random", "world_oracle"):
        get = lambda sub: next(x["accuracy"] for x in rows if x["method"] == r_ and x["cell"] == sub)
        print(f"{r_:22s}{get('ALL'):>9.4f}{get('AGREE'):>9.4f}{get('DISAGREE'):>10.4f}")
    print(f"\n{'primary':28s}{'delta':>9s}{'p':>9s}{'p_adj':>9s}  reading")
    for r in stats:
        print(f"{r['method'] + ' ' + r['cell']:28s}{r['delta']:>+9.4f}{r['p']:>9.4f}{r['p_adj']:>9.4f}  {r['reading']}")

    out = {"provenance": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
                         "prereg": PREREG, "smoke": smoke, "domain": DOMAIN, "margin": MARGIN,
                         "sources": sources,
                         "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                         "interpreter_seeds": list(c["seeds"]), "torch": torch.__version__},
           "rows": rows, "comparisons": stats, "diagnostic": diagnostic, "training": logs,
           "wall_seconds": time.time() - t0}
    Path(c["out"]).write_text(json.dumps(out, indent=1))
    print(f"wrote {c['out']} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(smoke="--smoke" in sys.argv[1:]))
