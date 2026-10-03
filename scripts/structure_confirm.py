#!/usr/bin/env python
"""Confirmatory: does the internals stream add to transfer?

Protocol frozen in docs/PREREG_STRUCTURE.md before this ran. Fresh populations.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal import ground_truth
from mint.causal.structure import TYPES, extract, heuristic_kind
from mint.domains import get_domain
from mint.eval.stats import comparison_table, min_attainable_p, run_comparisons
from mint.generative import position_first_order
from mint.generative.structure_model import StructureReader
from mint.models import TargetConfig, train_target
from mint.semantic.encoding import causal_pattern
from scripts.structure_transfer import balanced, pos_features  # reuse, do not fork

ARCH = {"family": "lm", "d_model": 48, "n_layers": 2, "n_heads": 4, "d_ff": 96}
N_TRAIN, EPOCHS, LR, N_PROBE = 8000, 60, 3e-3, 384
# Bumped whenever anything that shapes a cached pack changes. Caching is only
# safe if the key covers everything that determines the result; an incomplete
# key silently serves a stale artefact, which has bitten this project before.
PACK_VERSION = 1
CACHE = Path("results/cache/structure")
DOMAINS = ("interact", "witness")
ARMS = {"reader": {}, "reader_positions_only": {"use_units": False},
        "reader_units_only": {"use_positions": False}}
DECLARED = [{"metric": "balanced_accuracy", "method": "reader", "reference": "reader_positions_only"},
            {"metric": "balanced_accuracy", "method": "reader", "reference": "first_order_heuristic"},
            {"metric": "balanced_accuracy", "method": "reader_positions_only",
             "reference": "first_order_heuristic"},
            {"metric": "balanced_accuracy", "method": "reader_units_only", "reference": "random"}]


def _pack_key(domain_name, seed, n_focal, data_seed) -> str:
    payload = json.dumps({"v": PACK_VERSION, "domain": domain_name, "arch": ARCH,
                          "seed": seed, "spurious": 2.0, "n_train": N_TRAIN,
                          "epochs": EPOCHS, "lr": LR, "n_focal": n_focal,
                          "n_probe": N_PROBE, "data_seed": data_seed}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_one(domain_name, seed, n_focal, data_seed):
    """Train the target, measure its causal structure, encode it. Cached.

    Everything below is a deterministic function of the key, so reloading is
    identical to recomputing -- the cache changes the wall clock and nothing
    else. The expensive part is `extract`, which searches subsets by executing
    interventions.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{domain_name}-{_pack_key(domain_name, seed, n_focal, data_seed)}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = get_domain(domain_name)
    m = train_target(d, TargetConfig(domain_name, ARCH, seed, 2.0, n_train=N_TRAIN,
                                     epochs=EPOCHS, lr=LR),
                     np.random.default_rng(seed + 100 * data_seed)).model
    raw = d.sample(n_focal, np.random.default_rng(5000 + 97 * data_seed + seed)).raw
    gt = ground_truth(m, d, raw, d.sample(N_PROBE, np.random.default_rng(6000 + seed)).raw)
    st = extract(m, d, raw)
    ux, gx = causal_pattern(gt)
    fo = position_first_order(m, d, raw)
    pack = dict(unit_x=ux, global_x=gx, pos_x=pos_features(fo, st.candidates),
                y=st.kind, live=st.live, heur=heuristic_kind(st, fo))
    np.savez_compressed(path, **pack)
    return pack


def tensors(packs, u_max, p_max):
    pad = lambda a, n: np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, 0)))
    ux = np.concatenate([pad(p["unit_x"], u_max)[p["live"]] for p in packs])
    um = np.concatenate([np.pad(np.ones((int(p["live"].sum()), p["unit_x"].shape[1]), bool),
                                ((0, 0), (0, u_max - p["unit_x"].shape[1]))) for p in packs])
    px = np.concatenate([pad(p["pos_x"], p_max)[p["live"]] for p in packs])
    pm = np.concatenate([np.pad(np.ones((int(p["live"].sum()), p["pos_x"].shape[1]), bool),
                                ((0, 0), (0, p_max - p["pos_x"].shape[1]))) for p in packs])
    gx = np.concatenate([p["global_x"][p["live"]] for p in packs])
    y = np.concatenate([p["y"][p["live"]] for p in packs])
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    return (T(ux), T(um, torch.bool), T(px), T(pm, torch.bool), T(gx), T(y, torch.long))


def train_reader(tr, epochs=40, seed=0, **kw):
    torch.manual_seed(seed)
    m = StructureReader(**kw)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    ux, um, px, pm, gx, y = tr
    w = torch.bincount(y, minlength=len(TYPES)).float().clamp(min=1).pow(-0.5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.cross_entropy(m(ux[b], um[b], px[b], pm[b], gx[b]), y[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def per_model(m, packs, u_max, p_max) -> list[float]:
    out = []
    for p in packs:
        ux, um, px, pm, gx, y = tensors([p], u_max, p_max)
        out.append(balanced(m(ux, um, px, pm, gx).argmax(1).numpy(), y.numpy()))
    return out


def main(n_models: int = 24, n_focal: int = 80, seeds: int = 3, data_seed: int = 3) -> int:
    t0 = time.time()
    packs = {d: [build_one(d, s, n_focal, data_seed) for s in range(n_models)] for d in DOMAINS}
    u_max = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    p_max = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    fit = {d: packs[d][:half] for d in DOMAINS}
    held = {d: packs[d][half:] for d in DOMAINS}
    print(f"built in {time.time() - t0:.0f}s; {half} fitted / {n_models - half} held out per domain")

    rows, rng = [], np.random.default_rng(0)
    for src in DOMAINS:
        tgt = [d for d in DOMAINS if d != src][0]
        tr = tensors(fit[src], u_max, p_max)
        for arm, kw in ARMS.items():
            per_seed = [per_model(train_reader(tr, seed=sd, **kw), held[tgt], u_max, p_max)
                        for sd in range(seeds)]
            vals = np.mean(per_seed, axis=0)
            rows.append({"method": arm, "cell": f"{src}->{tgt}",
                         "per_model": [{"balanced_accuracy": float(v)} for v in vals],
                         "balanced_accuracy": float(vals.mean())})
            print(f"  {src}->{tgt} {arm}: {vals.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
        for nm, fn in (("first_order_heuristic",
                        lambda p: balanced(p["heur"][p["live"]], p["y"][p["live"]])),
                       ("random", lambda p: balanced(
                           rng.integers(0, len(TYPES), int(p["live"].sum())), p["y"][p["live"]])),
                       ("majority_of_source", None)):
            if nm == "majority_of_source":
                maj = int(np.bincount(np.concatenate([p["y"][p["live"]] for p in fit[src]])).argmax())
                vals = [balanced(np.full(int(p["live"].sum()), maj), p["y"][p["live"]])
                        for p in held[tgt]]
            else:
                vals = [fn(p) for p in held[tgt]]
            rows.append({"method": nm, "cell": f"{src}->{tgt}",
                         "per_model": [{"balanced_accuracy": float(v)} for v in vals],
                         "balanced_accuracy": float(np.mean(vals))})
        ceil = np.mean([per_model(train_reader(tensors(fit[tgt], u_max, p_max), seed=sd),
                                  held[tgt], u_max, p_max) for sd in range(seeds)], axis=0)
        rows.append({"method": "in_domain_ceiling", "cell": f"{src}->{tgt}",
                     "per_model": [{"balanced_accuracy": float(v)} for v in ceil],
                     "balanced_accuracy": float(ceil.mean())})

    cells = [f"{s}->{[d for d in DOMAINS if d != s][0]}" for s in DOMAINS]
    stats = run_comparisons(rows, DECLARED, 0.05, cells=cells)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    Path("results/structure_confirm.json").write_text(json.dumps(
        {"provenance": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                        "prereg": "docs/PREREG_STRUCTURE.md", "data_seed": data_seed,
                        "reader_seeds": list(range(seeds)), "n_models": n_models},
         "cells": cells, "rows": rows, "comparisons": stats,
         "wall_seconds": time.time() - t0}, indent=2, default=float))

    print(f"\n{'method':26s}" + "".join(c.rjust(22) for c in cells))
    print("-" * (26 + 22 * len(cells)))
    for nm in ("in_domain_ceiling", "reader", "reader_positions_only", "reader_units_only",
               "first_order_heuristic", "majority_of_source", "random"):
        print(f"{nm:26s}" + "".join(
            f"{[r for r in rows if r['method'] == nm and r['cell'] == c][0]['balanced_accuracy']:22.3f}"
            for c in cells))
    print(f"\ndeclared comparisons (Holm over {len(stats)}, smallest attainable p "
          f"{min_attainable_p(n_models - half):.5f})")
    print(comparison_table(stats))

    key = [x for x in stats if x["reference"] == "reader_positions_only"]
    beat = [x for x in key if x["reject"] and x["delta"] > 0]
    lost = [x for x in key if x["reject"] and x["delta"] < 0]
    heur = [x for x in stats if x["method"] == "reader" and x["reference"] == "first_order_heuristic"]
    units = [x for x in stats if x["method"] == "reader_units_only"]
    if beat and not lost and not any(u["reject"] and u["delta"] > 0 for u in units):
        v = "VOID - the full model beats positions-only while the internals stream does not beat random"
    elif beat and not lost:
        v = "INTERNALS ADD - reading the target's internal state contributes to transfer"
    elif any(h["reject"] and h["delta"] > 0 for h in heur):
        v = "MEASUREMENT TRANSFERS, INTERNALS DO NOT - the claim is about the first-order pattern"
    else:
        v = "FAIL - the exploratory transfer does not replicate"
    print(f"\nGATE: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
