#!/usr/bin/env python
"""EXPLORATORY: can the interpreter generate the whole causal claim?

The analytic layer is the instrument and is used as one -- its output is an input
here, and the competing generator is handed it too. The question is not whether a
learned model beats a derivative at measuring; it is whether it can turn a
measurement into a claim, which a derivative does not emit at all.

Both generators produce the same object -- kind, support, direction, magnitude --
and every clause is checked by executing the proposed intervention on the target.
The cache therefore stores the target's weights as well as the derived arrays:
without the model there is nothing to execute against.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal import ground_truth
from mint.causal.structure import TYPES, extract
from mint.domains import get_domain
from mint.generative import CompositeGenerator, formula_template, verify
from mint.generative.composite import Composite, bucket, support_effect
from mint.generative.task import position_first_order
from mint.models import TargetConfig, build_target, train_target
from mint.semantic.encoding import causal_pattern
from scripts.structure_transfer import pos_features

ARCH = {"family": "lm", "d_model": 48, "n_layers": 2, "n_heads": 4, "d_ff": 96}
DOMAINS = ("interact", "witness")
N_TRAIN, EPOCHS, LR, N_PROBE, PACK_V = 8000, 60, 3e-3, 384, 3
CACHE = Path("results/cache/composite")


def _labels(model, domain, raw, support, sd):
    """Sign and strength of *the support a claim proposes* -- the same quantity
    `verify()` scores against, via the same function.

    Derived on every load and deliberately not trusted from the cache. They were
    once written from `st.full_effect` while evaluation had already moved to the
    support's own effect; the two disagree on ~37% of live states (sign 5%,
    strength 35%), so every learned arm in `hybrid_confirm` and in the 96-cell
    scaling grid was fitted against a target it was never scored on. Recomputing
    here makes that class of drift impossible rather than merely fixed once.
    """
    a = support_effect(model, domain, raw, support.astype(bool)) / max(sd, 1e-6)
    return (a > 0).astype(np.int64), bucket(a).astype(np.int64)


def build_one(dom, seed, n_focal, data_seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps({"v": PACK_V, "d": dom, "a": ARCH, "s": seed, "n": N_TRAIN,
                                   "e": EPOCHS, "lr": LR, "f": n_focal, "p": N_PROBE,
                                   "ds": data_seed}, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"{dom}-{key}.pt"
    d = get_domain(dom)
    if path.exists():
        pack = torch.load(path, weights_only=False)
        model = build_target(d.spec, ARCH, seed)
        model.load_state_dict(pack.pop("state_dict"))
        pack["model"] = model.eval()
        pack["sign"], pack["strength"] = _labels(model, d, pack["raw"], pack["support"],
                                                 pack["logit_sd"])
        return pack
    m = train_target(d, TargetConfig(dom, ARCH, seed, 2.0, n_train=N_TRAIN, epochs=EPOCHS,
                                     lr=LR), np.random.default_rng(seed + 100 * data_seed)).model
    raw = d.sample(n_focal, np.random.default_rng(5000 + 97 * data_seed + seed)).raw
    gt = ground_truth(m, d, raw, d.sample(N_PROBE, np.random.default_rng(6000 + seed)).raw)
    st = extract(m, d, raw)
    ux, gx = causal_pattern(gt)
    fo = position_first_order(m, d, raw)
    sd = float(gt.probe.logit_std)
    sign, strength = _labels(m, d, raw, st.support, sd)
    pack = dict(unit_x=ux, global_x=gx, pos_x=pos_features(fo, st.candidates), fo=fo,
                cand=st.candidates, kind=st.kind, support=st.support,
                sign=sign, strength=strength,
                live=st.live.astype(bool), raw=raw, logit_sd=sd)
    torch.save({**pack, "state_dict": m.state_dict()}, path)
    pack["model"] = m.eval()
    return pack


def tensors(packs, u_max, p_max):
    pad3 = lambda a, n: np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, 0)))
    pad2 = lambda a, n: np.pad(a, ((0, 0), (0, n - a.shape[1])))
    L = [p["live"] for p in packs]
    T = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt)
    ones = lambda p, l, n, k: np.pad(np.ones((int(l.sum()), p[k].shape[1]), bool),
                                     ((0, 0), (0, n - p[k].shape[1])))
    return dict(
        unit_x=T([pad3(p["unit_x"], u_max)[l] for p, l in zip(packs, L)]),
        unit_mask=T([ones(p, l, u_max, "unit_x") for p, l in zip(packs, L)], torch.bool),
        pos_x=T([pad3(p["pos_x"], p_max)[l] for p, l in zip(packs, L)]),
        pos_mask=T([ones(p, l, p_max, "pos_x") for p, l in zip(packs, L)], torch.bool),
        global_x=T([p["global_x"][l] for p, l in zip(packs, L)]),
        cand=T([pad2(p["cand"], p_max)[l] for p, l in zip(packs, L)], torch.bool),
        kind=T([p["kind"][l] for p, l in zip(packs, L)], torch.long),
        support=T([pad2(p["support"], p_max)[l] for p, l in zip(packs, L)]),
        sign=T([p["sign"][l] for p, l in zip(packs, L)], torch.long),
        strength=T([p["strength"][l] for p, l in zip(packs, L)], torch.long))


def train(tr, epochs=50, seed=0, **kw):
    torch.manual_seed(seed)
    m = CompositeGenerator(**kw)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr["kind"])
    wk = torch.bincount(tr["kind"], minlength=len(TYPES)).float().clamp(min=1).pow(-0.5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            o = m(tr["unit_x"][b], tr["unit_mask"][b], tr["pos_x"][b], tr["pos_mask"][b],
                  tr["global_x"][b], tr["cand"][b])
            loss = (F.cross_entropy(o["kind"], tr["kind"][b], weight=wk)
                    + F.cross_entropy(o["sign"], tr["sign"][b])
                    + F.cross_entropy(o["strength"], tr["strength"][b])
                    + (F.binary_cross_entropy_with_logits(o["support"], tr["support"][b],
                                                          reduction="none")
                       * tr["cand"][b]).sum() / tr["cand"][b].sum().clamp(min=1))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def generate(m, pack, u_max, p_max) -> Composite:
    t = tensors([pack], u_max, p_max)
    o = m(t["unit_x"], t["unit_mask"], t["pos_x"], t["pos_mask"], t["global_x"], t["cand"])
    live = pack["live"]
    n, T = len(live), pack["pos_x"].shape[1]
    idx = np.nonzero(live)[0]
    kind = np.zeros(n, int); sign = np.zeros(n, int); strength = np.zeros(n, int)
    sup = np.zeros((n, T), bool)
    kind[idx] = o["kind"].argmax(1).numpy()
    sign[idx] = o["sign"].argmax(1).numpy()
    strength[idx] = o["strength"].argmax(1).numpy()
    p = torch.sigmoid(o["support"]).numpy()[:, :T]
    ch = (p > 0.5) & pack["cand"][idx]
    empty = ~ch.any(1)
    if empty.any():
        ch[empty, np.where(pack["cand"][idx], p, -1).argmax(1)[empty]] = True
    sup[idx] = ch
    return Composite(kind, sup, sign, strength, live)


def main(n_models: int = 12, n_focal: int = 80, seeds: int = 2, data_seed: int = 4) -> int:
    t0 = time.time()
    packs = {d: [build_one(d, s, n_focal, data_seed) for s in range(n_models)] for d in DOMAINS}
    u_max = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    p_max = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    print(f"built in {time.time() - t0:.0f}s", flush=True)

    cols = ["type", "support_exact", "sign", "strength", "composition",
            "verified_sign", "verified_strength"]
    for src in DOMAINS:
        tgt = [x for x in DOMAINS if x != src][0]
        fit, held = packs[src][:half], packs[tgt][half:]
        tr = tensors(fit, u_max, p_max)
        maj = int(np.bincount(tr["kind"].numpy()).argmax())
        d = get_domain(tgt)
        agg = {k: [] for k in ("generator", "formula_template", "oracle")}
        for sd in range(seeds):
            m = train(tr, seed=sd)
            for p in held:
                gtc = Composite(p["kind"], p["support"].astype(bool), p["sign"],
                                p["strength"], p["live"])
                agg["generator"].append(verify(p["model"], d, p["raw"],
                                               generate(m, p, u_max, p_max), p["logit_sd"], gtc))
                if sd == 0:
                    agg["formula_template"].append(verify(
                        p["model"], d, p["raw"],
                        formula_template(p["model"], d, p["raw"], p["fo"], p["cand"],
                                         p["live"], p["logit_sd"], maj), p["logit_sd"], gtc))
                    agg["oracle"].append(verify(p["model"], d, p["raw"], gtc, p["logit_sd"], gtc))
        print(f"\n### trained on {src}, generating for {tgt}   "
              f"({half} held-out targets, {seeds} seeds)")
        print(f"{'method':20s}" + "".join(c.rjust(17) for c in cols))
        print("-" * (20 + 17 * len(cols)))
        for k in ("oracle", "generator", "formula_template"):
            v = [r for r in agg[k] if r]
            print(f"{k:20s}" + "".join(f"{np.mean([r[c] for r in v]):17.3f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
