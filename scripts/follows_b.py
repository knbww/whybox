#!/usr/bin/env python
"""Does the interpreter follow the model, or the world?

Protocol frozen in `docs/PREREG_FOLLOWS_B.md` before this ran.

Everything this project measured before 2026-09-13 was scored against
`gt.decisive`, the *world's* decisive factor, which does not depend on the target
model at all. This scores against `argmax abs(gt.factor_total)` -- B's own
response to neutralising each factor -- and reports the states where the two part
company separately, because that is the only place the difference between
interpreting a model and predicting a simulator is visible.

The interpreter is never handed a list of causes. It emits one score per input
position of B; the name is recovered afterwards from whose carriers those
positions are.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mint.domains import get_domain
from mint.encoding.model_state import tied_rank
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from mint.generative.task import _contrast
from mint.generative.task import position_first_order as _lm_position_first_order
from mint.semantic.encoding import causal_pattern

DOMAIN, MARGIN, SENSITIVITY = "skirmish", 0.10, (0.15, 0.25)

# The target populations every committed result in this line was computed on,
# pinned by path and content hash. Choosing `sorted(glob(...))[0]` changed the
# dataset silently whenever a new cache file happened to sort earlier, and for
# `entail` it picked a stale file with no `gt.first_order` that crashes on load.
CACHE = {
    "skirmish": ("results/cache/skirmish-mlp-6d7fca3aed.pt", "1f8d8e7ef6d4edd6"),
    "clinic": ("results/cache/clinic-mlp-6326cb5796.pt", "66d838d8f3f1935a"),
    "entail": ("results/cache/entail-lm-b75d8fee9e.pt", "829dc1bb57aa15b7"),
}


def load_population(domain: str, n_models: int = 24):
    """The pinned bundles for `domain`, refusing a file whose contents changed."""
    import hashlib
    path, want = CACHE[domain]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if not got.startswith(want):
        raise RuntimeError(f"{path}: sha256 {got[:16]} does not match the pinned {want}")
    return torch.load(path, weights_only=False)[:n_models], path, got
# The family exactly as docs/PREREG_FOLLOWS_B.md declares it: six comparisons, not
# the 3x3 cross-product `run_comparisons` would build. Holm spans these six only.
FAMILY = [("constant", "DISAGREE"), ("world_oracle", "DISAGREE"), ("random", "DISAGREE"),
          ("constant", "AGREE"), ("random", "AGREE"), ("constant", "ALL")]


def pos_first_order(model, domain, raw: np.ndarray) -> np.ndarray:
    """(N, T) first-order effect of moving each input position to the reference.

    Two branches because the gradient is taken with respect to different things:
    a tabular target's inputs are continuous and differentiable directly, while a
    language target's are token ids and the gradient goes through the embedding.
    `mint.generative.task.position_first_order` is the latter and is reused
    unchanged.
    """
    if domain.spec.kind in ("sequence", "language"):
        return _lm_position_first_order(model, domain, raw)
    ctx = torch.as_tensor(domain.to_model_input(raw), dtype=torch.float32).requires_grad_(True)
    ref = torch.as_tensor(domain.to_model_input(_contrast(domain, raw)), dtype=torch.float32)
    g, = torch.autograd.grad(model(ctx)[0].sum(), ctx)
    return (g * (ref - ctx)).detach().numpy()


def pos_features(fo: np.ndarray) -> np.ndarray:
    """(N, T, 5) from B's per-position sensitivity alone.

    No candidate mask: that is derived from the human's counterfactual recipe and
    would tell the interpreter which positions are in play before it has read
    anything.
    """
    a = np.abs(fo)
    scale = np.maximum(a.max(1, keepdims=True), 1e-9)
    return np.nan_to_num(np.stack([
        fo / scale, a / scale,
        np.stack([tied_rank(a[i]) for i in range(len(fo))]),
        a / (a.sum(1, keepdims=True) + 1e-9), np.sign(fo),
    ], -1)).astype(np.float32)


def carriers_in(spec, width: int) -> list[list[int]]:
    """Carrier indices that exist in the model's input space.

    A language target's `to_model_input` drops the last raw column -- the token it
    predicts -- so the scored width is one less than `spec.n_inputs`. Carrier
    indices are given in the raw space; any that fall outside the scored space
    cannot be pointed at and are dropped here rather than silently misaligning.
    """
    out = []
    for f in spec.factors:
        kept = [c for c in f.carriers if c < width]
        if not kept:
            # Aliasing a cause with no visible carrier onto position 0 would make it
            # silently unnameable; refuse instead.
            raise ValueError(f"{f.name}: every carrier {f.carriers} lies outside the "
                             f"model's input width {width}")
        out.append(kept)
    return out


def recover_name(scores: np.ndarray, spec) -> np.ndarray:
    """(N,) cause index: whose carriers did the pointer score highest, on average."""
    car = carriers_in(spec, scores.shape[1])
    per = np.stack([scores[:, c].mean(1) for c in car], 1)
    return per.argmax(1)


def _enc(d_in, d_model, n_layers, n_heads, dropout):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return (nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_model)),
            nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False))


class Pointer(nn.Module):
    """Reads B's internals, points at B's input positions. No list of causes."""

    def __init__(self, d_unit, d_pos, d_global, d_model=96, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.unit_in, self.unit_enc = _enc(d_unit, d_model, n_layers, n_heads, dropout)
        self.pos_in, self.pos_enc = _enc(d_pos, d_model, n_layers, n_heads, dropout)
        self.ctx = nn.Sequential(nn.Linear(2 * d_model + d_global, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, 1))

    def forward(self, unit_x, unit_mask, pos_x, global_x):
        hu = self.unit_enc(self.unit_in(unit_x), src_key_padding_mask=~unit_mask)
        u = (hu * unit_mask[..., None]).sum(1) / unit_mask.sum(1, keepdim=True).clamp(min=1)
        hp = self.pos_enc(self.pos_in(pos_x))
        c = self.ctx(torch.cat([u, hp.mean(1), global_x], -1))
        z = torch.cat([hp, c[:, None, :].expand(-1, hp.shape[1], -1)], -1)
        return self.head(z).squeeze(-1)


def prepare(bundle, domain) -> dict:
    gt = bundle.gt
    ux, gx = causal_pattern(gt)
    fo = pos_first_order(bundle.trained.model, domain, bundle.focal_raw)
    a = np.abs(gt.factor_total) / max(gt.probe.logit_std, 1e-9)
    s = np.sort(a, axis=1)
    y_model = a.argmax(1)                      # what THIS model relies on
    return dict(unit_x=ux, global_x=gx, pos_x=pos_features(fo), fo=fo,
                y=y_model, world=gt.decisive, margin=s[:, -1] - s[:, -2])


def tensors(packs, u_max, spec, thr):
    keep = [p["margin"] > thr for p in packs]
    pad = lambda a, n: np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, 0)))
    ux = np.concatenate([pad(p["unit_x"], u_max)[k] for p, k in zip(packs, keep)])
    um = np.concatenate([np.pad(np.ones((int(k.sum()), p["unit_x"].shape[1]), bool),
                                ((0, 0), (0, u_max - p["unit_x"].shape[1])))
                         for p, k in zip(packs, keep)])
    px = np.concatenate([p["pos_x"][k] for p, k in zip(packs, keep)])
    gx = np.concatenate([p["global_x"][k] for p, k in zip(packs, keep)])
    y = np.concatenate([p["y"][k] for p, k in zip(packs, keep)])
    car = carriers_in(spec, px.shape[1])
    carr = np.zeros((len(y), px.shape[1]), np.float32)
    for i, f in enumerate(y):
        carr[i, car[f]] = 1.0
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    return T(ux), T(um, torch.bool), T(px), T(gx), T(carr), y


def train_pointer(tr, spec, seed=0, epochs=40):
    torch.manual_seed(seed)
    ux, um, px, gx, carr, _ = tr
    m = Pointer(ux.shape[-1], px.shape[-1], gx.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(carr), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(m(ux[b], um[b], px[b], gx[b]), carr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def named(m, pack, spec, u_max, thr) -> np.ndarray:
    tr = tensors([pack], u_max, spec, thr)
    return recover_name(m(*tr[:4]).numpy(), spec)


def arms(pack, m, spec, u_max, thr, maj, rng) -> dict[str, np.ndarray]:
    k = pack["margin"] > thr
    n = int(k.sum())
    return {
        "interpreter": named(m, pack, spec, u_max, thr),
        "world_oracle": pack["world"][k],
        "first_order_pointer": recover_name(np.abs(pack["fo"][k]), spec),
        "constant": np.full(n, maj),
        "random": rng.integers(0, spec.n_factors, n),
    }


def main(n_models: int = 24, seeds: int = 3) -> int:
    torch.set_num_threads(4)   # the trainer barely scales with threads; see CLAUDE.md
    t0 = time.time()
    d = get_domain(DOMAIN); spec = d.spec
    bundles, cache_path, cache_sha = load_population(DOMAIN, n_models)
    packs = [prepare(b, d) for b in bundles]
    u_max = max(p["unit_x"].shape[1] for p in packs)
    fit, held = packs[:12], packs[12:]
    print(f"{DOMAIN}: {len(fit)} моделей на обучение, {len(held)} отложено "
          f"({time.time() - t0:.0f}s)", flush=True)

    out_rows, out = [], {}
    for thr in (MARGIN, *SENSITIVITY):
        tr = tensors(fit, u_max, spec, thr)
        maj = int(np.bincount(tr[5], minlength=spec.n_factors).argmax())
        models = []
        for sd in range(seeds):
            models.append(train_pointer(tr, spec, seed=sd))
            print(f"    порог {thr}, seed {sd} обучен ({time.time() - t0:.0f}s)", flush=True)
        rng = np.random.default_rng(0)
        per = {}
        for p in held:
            k = p["margin"] > thr
            y, agree = p["y"][k], (p["world"][k] == p["y"][k])
            preds = [arms(p, m, spec, u_max, thr, maj, rng) for m in models]
            for nm in preds[0]:
                acc = np.mean([(pr[nm] == y) for pr in preds], axis=0)   # over seeds
                for sub, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree),
                                 ("DISAGREE", ~agree)):
                    per.setdefault((nm, sub), []).append(
                        float(acc[msk].mean()) if msk.sum() else np.nan)
        tag = "" if thr == MARGIN else f" [sens {thr}]"
        for (nm, sub), vals in per.items():
            v = [x for x in vals if np.isfinite(x)]
            row = {"method": nm, "cell": sub + tag, "n_models": len(v),
                   "per_model": [{"accuracy": x} for x in v],
                   "accuracy": float(np.mean(v))}
            (out_rows if thr == MARGIN else out.setdefault("sensitivity", [])).append(row)
        n_dis = sum(int(((p["margin"] > thr) & (p["world"] != p["y"])).sum()) for p in held)
        print(f"  порог {thr}: {n_dis} расходящихся состояний на {len(held)} моделях "
              f"({time.time() - t0:.0f}s)", flush=True)
        for (nm, sub), vals in sorted(per.items()):
            v = [x for x in vals if np.isfinite(x)]
            print(f"     {nm:22s}{sub:10s}{np.mean(v):.4f}")

    by = {(r["method"], r["cell"]): r for r in out_rows}
    stats, pv = [], []
    for ref, sub in FAMILY:
        a = np.array([m["accuracy"] for m in by[("interpreter", sub)]["per_model"]])
        b = np.array([m["accuracy"] for m in by[(ref, sub)]["per_model"]])
        delta, p = paired_sign_flip(a, b)
        stats.append({"metric": "accuracy", "method": "interpreter", "reference": ref,
                      "cell": sub, "n_models": len(a), "delta": delta,
                      "a": float(a.mean()), "b": float(b.mean())})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
    print(f"\n{'comparison':40s}{'subset':11s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  sig")
    print("-" * 82)
    for r in stats:
        print(f"interpreter vs {r['reference']:25s}{r['cell']:11s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  {('+' if r['delta'] > 0 else '-') if r['reject'] else ''}")
    prim = [c for c in stats if c["cell"] == "DISAGREE"
            and c["reference"] in ("constant", "world_oracle")]
    void = any(c["cell"] == "DISAGREE" and c["reference"] == "random" and not c["reject"]
               for c in stats)
    ok = all(c["reject"] and c["delta"] > 0 for c in prim) and len(prim) == 2
    gate = "VOID" if ok and void else "CONFIRMED" if ok else "FAIL"
    print(f"\nGATE: {gate}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out |= {"provenance": {"commit": git("rev-parse", "HEAD"),
                           "dirty": bool(git("status", "--porcelain")),
                           "prereg": "docs/PREREG_FOLLOWS_B.md", "domain": DOMAIN,
                           "margin": MARGIN, "cache": cache_path, "cache_sha256": cache_sha,
                           "n_models": n_models,
                           "seeds": list(range(seeds)), "torch": torch.__version__},
            "rows": out_rows, "comparisons": stats, "gate": gate,
            "wall_seconds": time.time() - t0}
    with open("results/follows_b.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/follows_b.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
