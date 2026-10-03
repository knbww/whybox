#!/usr/bin/env python
"""Does the *kind* of causal structure transfer between models and tasks?

EXPLORATORY. Trains a reader on one interaction domain and tests it on held-out
models of that domain and on a second domain whose surface has nothing in common
-- different vocabulary, different length, cues at random positions -- but which
carries the same four structures.

Declared before running: the competitors are the first-order heuristic, the
source domain's majority class carried over, random, and an in-domain ceiling.
Metric is balanced accuracy over five types (chance 0.20).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal import ground_truth
from mint.causal.structure import TYPES, extract, heuristic_kind
from mint.domains import get_domain
from mint.encoding.model_state import tied_rank
from mint.generative import position_first_order
from mint.generative.structure_model import StructureReader
from mint.models import TargetConfig, train_target
from mint.semantic.encoding import causal_pattern

ARCH = {"family": "lm", "d_model": 48, "n_layers": 2, "n_heads": 4, "d_ff": 96}
DOMAINS = ("interact", "witness")


def pos_features(fo: np.ndarray, cand: np.ndarray) -> np.ndarray:
    n, t = fo.shape
    a = np.abs(fo)
    scale = np.maximum(a.max(1, keepdims=True), 1e-9)
    share = a / (a.sum(1, keepdims=True) + 1e-9)
    rank = np.stack([tied_rank(a[i]) for i in range(n)])
    x = np.stack([fo / scale, a / scale, rank, share, cand.astype(float), np.sign(fo),
                  np.repeat(cand.sum(1, keepdims=True) / t, t, 1)], -1)
    return np.nan_to_num(x).astype(np.float32)


def build(domain_name: str, seeds, n_focal: int, n_probe: int = 384):
    d = get_domain(domain_name)
    packs = []
    for s in seeds:
        m = train_target(d, TargetConfig(domain_name, ARCH, s, 2.0, n_train=8000,
                                         epochs=60, lr=3e-3), np.random.default_rng(s)).model
        raw = d.sample(n_focal, np.random.default_rng(900 + s)).raw
        gt = ground_truth(m, d, raw, d.sample(n_probe, np.random.default_rng(950 + s)).raw)
        st = extract(m, d, raw)
        ux, gx = causal_pattern(gt)
        px = pos_features(position_first_order(m, d, raw), st.candidates)
        packs.append(dict(unit_x=ux, global_x=gx, pos_x=px, y=st.kind, live=st.live,
                          heur=heuristic_kind(st, position_first_order(m, d, raw))))
    return packs


def stack(packs, u_max, p_max):
    def pad(a, n):
        return np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, 0)))
    ux = np.concatenate([pad(p["unit_x"], u_max)[p["live"]] for p in packs])
    um = np.concatenate([np.pad(np.ones((int(p["live"].sum()), p["unit_x"].shape[1]), bool),
                                ((0, 0), (0, u_max - p["unit_x"].shape[1]))) for p in packs])
    px = np.concatenate([pad(p["pos_x"], p_max)[p["live"]] for p in packs])
    pm = np.concatenate([np.pad(np.ones((int(p["live"].sum()), p["pos_x"].shape[1]), bool),
                                ((0, 0), (0, p_max - p["pos_x"].shape[1]))) for p in packs])
    gx = np.concatenate([p["global_x"][p["live"]] for p in packs])
    y = np.concatenate([p["y"][p["live"]] for p in packs])
    h = np.concatenate([p["heur"][p["live"]] for p in packs])
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d)
    return (T(ux), T(um, torch.bool), T(px), T(pm, torch.bool), T(gx), T(y, torch.long), h)


def balanced(pred, y) -> float:
    return float(np.mean([float((pred[y == c] == c).mean()) for c in np.unique(y)]))


def train_reader(tr, epochs=40, seed=0, use_units=True, use_positions=True):
    torch.manual_seed(seed)
    m = StructureReader(use_units=use_units, use_positions=use_positions)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    ux, um, px, pm, gx, y, _ = tr
    cnt = torch.bincount(y, minlength=len(TYPES)).float()
    w = cnt.clamp(min=1).pow(-0.5)
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
def predict(m, ds) -> np.ndarray:
    ux, um, px, pm, gx, _, _ = ds
    return m(ux, um, px, pm, gx).argmax(1).numpy()


def main(n_models: int = 10, n_focal: int = 80, seeds: int = 2) -> int:
    t0 = time.time()
    packs = {d: build(d, range(n_models), n_focal) for d in DOMAINS}
    u_max = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    p_max = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    fit = {d: stack(packs[d][:half], u_max, p_max) for d in DOMAINS}
    held = {d: stack(packs[d][half:], u_max, p_max) for d in DOMAINS}
    print(f"built in {time.time() - t0:.0f}s | " +
          " ".join(f"{d}: fit {len(fit[d][5])} held {len(held[d][5])}" for d in DOMAINS))

    rows = []
    for src in DOMAINS:
        tgt = [d for d in DOMAINS if d != src][0]
        maj = int(np.bincount(fit[src][5].numpy()).argmax())
        for arm, kw in (("reader", {}), ("reader_units_only", {"use_positions": False}),
                        ("reader_positions_only", {"use_units": False})):
            got = {d: [] for d in DOMAINS}
            for sd in range(seeds):
                m = train_reader(fit[src], seed=sd, **kw)
                for d in DOMAINS:
                    got[d].append(balanced(predict(m, held[d]), held[d][5].numpy()))
            for d in DOMAINS:
                rows.append((src, d, arm, float(np.mean(got[d]))))
        for d in DOMAINS:
            y = held[d][5].numpy()
            rows.append((src, d, "first_order_heuristic", balanced(held[d][6], y)))
            rows.append((src, d, f"majority_of_{src}", balanced(np.full_like(y, maj), y)))
            rng = np.random.default_rng(0)
            rows.append((src, d, "random", balanced(rng.integers(0, len(TYPES), len(y)), y)))
        ceil = []
        for sd in range(seeds):
            m = train_reader(fit[tgt], seed=sd)
            ceil.append(balanced(predict(m, held[tgt]), held[tgt][5].numpy()))
        rows.append((src, tgt, "in_domain_ceiling", float(np.mean(ceil))))

    for src in DOMAINS:
        print(f"\n### trained on {src}   (balanced accuracy over {len(TYPES)} types, chance 0.200)")
        print(f"{'method':26s}" + "".join(f"{d:>20s}" for d in DOMAINS))
        print("-" * (26 + 20 * len(DOMAINS)))
        names = ["in_domain_ceiling", "reader", "reader_units_only", "reader_positions_only",
                 "first_order_heuristic", f"majority_of_{src}", "random"]
        for nm in names:
            cells = []
            for d in DOMAINS:
                v = [r[3] for r in rows if r[0] == src and r[1] == d and r[2] == nm]
                cells.append(f"{v[0]:20.3f}" if v else " " * 20)
            tag = nm + ("  [TRANSFER]" if nm == "reader" else "")
            print(f"{tag:26s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
