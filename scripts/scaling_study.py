#!/usr/bin/env python
"""EXPLORATORY scaling study. Protocol frozen in docs/PROTOCOL_SCALING.md.

Same task, same metrics, same interface as the confirmatory hybrid run. Two
things vary and nothing else: how much capacity the interpreter has, and how much
causal experience it was given. The question is not whether it is right -- it is
whether being bigger, or having seen more models, predictably makes it righter.

The evaluation is exact but cheap. Under the frozen interface the carrier is
always the instrument's, so the intervention a claim proposes is the *same*
intervention in every cell of the grid, for a given held-out target. Its executed
effect is therefore computed once per target and reused; scoring a claim after
that is arithmetic. `phase=check` verifies this path reproduces the confirmatory
numbers before any of it is believed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal.structure import TYPES
from mint.domains import get_domain
from mint.generative import JointGenerator, formula_template
from mint.generative.composite import bucket, support_effect
from scripts.composite_corrected import corrected_truth, train_joint
from scripts.composite_explore import DOMAINS, build_one, tensors

# (d_model, n_layers, n_heads) -> 0.043M, 0.263M, 0.585M (current), 3.20M, 16.29M
CAPS = [(32, 1, 4), (64, 2, 4), (96, 2, 4), (192, 3, 4), (384, 4, 8)]
GATED = (768, 6, 8)                      # 93.4M, only if a trend is visible
DATA = [3, 6, 12, 24]                    # fitted target models: 25%, 50%, 100%, 2x
FIT_POOL = list(range(0, 12)) + list(range(24, 36))
HELD = list(range(12, 24))               # identical to the confirmatory run
STEPS, BATCH, SEEDS = 750, 64, 3
LRS = [3e-4, 1e-3, 3e-3]
METRICS = ["structure", "sign", "magnitude", "executed_validity", "whole_statement"]
OUT = Path("results/scaling_study.json")
SHARDS = Path("results/scaling")


def nparams(cap) -> int:
    return sum(p.numel() for p in JointGenerator(*cap).parameters())


def train_steps(tr, cap, steps, lr, seed, lam=1.0):
    """Identical objective to the confirmatory arm; a step budget instead of epochs.

    Fixing steps rather than epochs is what keeps the data axis about data: with
    epochs fixed, a quarter of the data is also a quarter of the gradient updates
    and the two causes could not be told apart.
    """
    torch.manual_seed(seed)
    m = JointGenerator(*cap)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    n = len(tr["kind"])
    wk = torch.bincount(tr["kind"], minlength=len(TYPES)).float().clamp(min=1).pow(-0.5)
    g = torch.Generator().manual_seed(seed)
    m.train()
    for _ in range(steps):
        b = torch.randint(0, n, (min(BATCH, n),), generator=g)
        opt.zero_grad()
        loss = _loss(m, tr, b, wk, lam)
        loss.backward()
        opt.step()
    return m.eval()


def _loss(m, tr, b, wk, lam):
    args = dict(unit_x=tr["unit_x"][b], unit_mask=tr["unit_mask"][b], pos_x=tr["pos_x"][b],
                pos_mask=tr["pos_mask"][b], global_x=tr["global_x"][b], cand=tr["cand"][b])
    forced = m(**args, force_kind=tr["kind"][b], force_carrier=tr["support"][b])
    free = m(**args)
    sup = (F.binary_cross_entropy_with_logits(forced["support"], tr["support"][b],
                                              reduction="none")
           * tr["cand"][b]).sum() / tr["cand"][b].sum().clamp(min=1)
    clause = (F.cross_entropy(forced["kind"], tr["kind"][b], weight=wk) + sup
              + F.cross_entropy(forced["sign"], tr["sign"][b])
              + F.cross_entropy(forced["strength"], tr["strength"][b]))
    joint = (F.cross_entropy(free["sign"], tr["sign"][b])
             + F.cross_entropy(free["strength"], tr["strength"][b]))
    return clause + lam * joint


def bench(pack, domain, u, q) -> dict:
    """Per held-out target: everything a score depends on that the interpreter does not."""
    m, raw = pack["model"], pack["raw"]
    sd = max(pack["logit_sd"], 1e-6)
    f = formula_template(m, domain, raw, pack["fo"], pack["cand"], pack["live"], sd,
                         majority_type=0, execute_sign_strength=False)
    gt = corrected_truth(pack, domain)
    idx = np.nonzero(pack["live"])[0]
    dex = support_effect(m, domain, raw, f.support)[idx] / sd
    t = tensors([pack], u, q)
    c = np.zeros((len(idx), t["pos_x"].shape[1]), np.float32)
    c[:, :f.support.shape[1]] = f.support[idx]
    sup_ok = np.array([set(np.nonzero(f.support[i])[0]) == set(np.nonzero(gt.support[i])[0])
                       for i in idx], float)
    return dict(t=t, carrier=torch.as_tensor(c), truth=(gt.kind[idx], gt.sign[idx],
                gt.strength[idx]), sup_ok=sup_ok, exec_sign=(dex > 0).astype(int),
                exec_str=bucket(dex), inst=(f.sign[idx], f.strength[idx]))


def score(b, kind, sign, strength) -> dict:
    tk, ts, tm = b["truth"]
    ok_t, ok_s = (kind == tk).astype(float), (sign == ts).astype(float)
    ok_m = (strength == tm).astype(float)
    ev = ((sign == b["exec_sign"]) & (strength == b["exec_str"])).astype(float)
    return {"structure": float(ok_t.mean()), "sign": float(ok_s.mean()),
            "magnitude": float(ok_m.mean()), "executed_validity": float(ev.mean()),
            "whole_statement": float((ok_t * b["sup_ok"] * ok_s * ok_m).mean())}


@torch.no_grad()
def claim(m, b):
    o = m(b["t"]["unit_x"], b["t"]["unit_mask"], b["t"]["pos_x"], b["t"]["pos_mask"],
          b["t"]["global_x"], b["t"]["cand"], force_carrier=b["carrier"])
    return (o["kind"].argmax(1).numpy(), o["sign"].argmax(1).numpy(),
            o["strength"].argmax(1).numpy())


N_FOCAL, DATA_SEED = 80, 5


def load(dom, seeds) -> dict:
    """Only the packs a shard actually needs -- 120 processes should not each hold all."""
    return {s: build_one(dom, s, N_FOCAL, DATA_SEED) for s in seeds}


def widths():
    """Padding widths are a property of the domain and the target architecture, not of
    the seed -- verified across every cached pack. Taking them from one pack per domain
    gives every shard the identical input shape without loading the rest; each pack that
    is then used is checked against them."""
    one = {d: build_one(d, HELD[0], N_FOCAL, DATA_SEED) for d in DOMAINS}
    return (max(p["unit_x"].shape[1] for p in one.values()),
            max(p["pos_x"].shape[1] for p in one.values()))


def check_widths(packs, u, q):
    for p in packs.values():
        assert p["unit_x"].shape[1] <= u and p["pos_x"].shape[1] <= q, "width drift"


def other(src: str) -> str:
    return [d for d in DOMAINS if d != src][0]


def held_benches(tgt, u, q):
    """Cached: benches do not depend on the interpreter, so without this the same
    greedy carrier search would be recomputed in every one of the shards."""
    f = SHARDS / f"bench-{tgt}-{u}x{q}.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    packs = load(tgt, HELD)
    check_widths(packs, u, q)
    bs = [bench(packs[s], get_domain(tgt), u, q) for s in HELD]
    torch.save(bs, f)
    return bs


def fitted(src, n, u, q):
    packs = load(src, FIT_POOL[:n])
    check_widths(packs, u, q)
    return tensors([packs[s] for s in FIT_POOL[:n]], u, q)


def main(phase: str = "run", *rest) -> int:
    t0 = time.time()
    import os
    torch.set_num_threads(int(os.environ.get("MINT_THREADS", "1")))
    SHARDS.mkdir(parents=True, exist_ok=True)
    u, q = widths()

    if phase == "check":
        # the fast evaluation path must reproduce results/frozen/hybrid_confirm.json
        ref = {r["cell"]: r["executed_validity"] for r in
               json.load(open("results/frozen/hybrid_confirm.json"))["rows"]
               if r["method"] == "hybrid"}
        for src in ([rest[0]] if rest else DOMAINS):
            tgt = other(src)
            bs, tr = held_benches(tgt, u, q), fitted(src, 12, u, q)
            got = []
            for sd in range(SEEDS):      # train once per seed, not once per held-out target
                m = train_joint(tr, seed=sd)
                got.append(np.mean([score(b, *claim(m, b))["executed_validity"] for b in bs]))
            r = ref[f"{src}->{tgt}"]
            print(f"{src}->{tgt}: fast path {np.mean(got):.4f}  frozen {r:.4f}  "
                  f"delta {np.mean(got) - r:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        return 0

    if phase == "lr":                      # one (direction, capacity, rate), held-in only
        src, ci, li = rest[0], int(rest[1]), int(rest[2])
        tr = fitted(src, 10, u, q)
        packs = load(src, FIT_POOL[10:12])
        va = tensors([packs[s] for s in FIT_POOL[10:12]], u, q)
        wk = torch.bincount(tr["kind"], minlength=len(TYPES)).float().clamp(min=1).pow(-0.5)
        m = train_steps(tr, CAPS[ci], STEPS, LRS[li], 0)
        with torch.no_grad():
            v = float(_loss(m, va, torch.arange(len(va["kind"])), wk, 1.0))
        (SHARDS / f"lrscan-{src}-{ci}-{li}.json").write_text(
            json.dumps({"src": src, "ci": ci, "lr": LRS[li], "val": v}))
        print(f"{src} cap{ci} lr {LRS[li]:g} val {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return 0

    if phase == "lrpick":
        for src in DOMAINS:
            for ci in range(len(CAPS)):
                sc = [json.loads(f.read_text())
                      for f in SHARDS.glob(f"lrscan-{src}-{ci}-*.json")]
                if len(sc) < len(LRS):       # a rung not yet swept is simply not picked
                    continue
                best = min(sc, key=lambda d: d["val"])
                (SHARDS / f"lr-{src}-{ci}.json").write_text(json.dumps(
                    {"src": src, "cap": list(CAPS[ci]), "params": nparams(CAPS[ci]),
                     "lr": best["lr"], "val": best["val"],
                     "scan": {str(d["lr"]): d["val"] for d in sc}}))
                print(f"{src} {nparams(CAPS[ci]) / 1e6:7.3f}M -> lr {best['lr']:g}   "
                      + " ".join(f"{d['lr']:g}:{d['val']:.4f}"
                                 for d in sorted(sc, key=lambda d: d["lr"])), flush=True)
        return 0

    if phase == "shard":                   # one (direction, capacity, data level, seed)
        src, ci, di, sd = rest[0], int(rest[1]), int(rest[2]), int(rest[3])
        tgt, nd = other(src), DATA[di]
        lr = json.loads((SHARDS / f"lr-{src}-{ci}.json").read_text())["lr"]
        m = train_steps(fitted(src, nd, u, q), CAPS[ci], STEPS, lr, sd)
        pm = [score(b, *claim(m, b)) for b in held_benches(tgt, u, q)]
        (SHARDS / f"cell-{src}-{ci}-{di}-{sd}.json").write_text(json.dumps(
            {"cell": f"{src}->{tgt}", "params": nparams(CAPS[ci]), "cap": list(CAPS[ci]),
             "n_fitted": nd, "lr": lr, "seed": sd, "per_model": pm}, default=float))
        print(f"{src}->{tgt} {nparams(CAPS[ci]) / 1e6:7.3f}M n={nd:2d} seed{sd} "
              + " ".join(f"{k[:4]} {np.mean([x[k] for x in pm]):.3f}" for k in METRICS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        return 0

    if phase == "baseline":                # no capacity axis, no data axis: one number
        out = []
        for src in DOMAINS:
            tgt = other(src)
            maj = int(np.bincount(fitted(src, 12, u, q)["kind"].numpy()).argmax())
            pm = [score(b, np.full(len(b["sup_ok"]), maj), *b["inst"])
                  for b in held_benches(tgt, u, q)]
            out.append({"cell": f"{src}->{tgt}", "params": 0, "cap": None,
                        "n_fitted": None, "lr": None, "analytic": True, "per_model": pm,
                        **{k: float(np.mean([x[k] for x in pm])) for k in METRICS}})
            print(f"{src}->{tgt} ANALYTIC "
                  + " ".join(f"{k[:4]} {out[-1][k]:.3f}" for k in METRICS), flush=True)
        (SHARDS / "baseline.json").write_text(json.dumps(out, default=float))
        return 0

    if phase == "merge":
        by, lrs = {}, {}
        for f in sorted(SHARDS.glob("cell-*.json")):
            d = json.loads(f.read_text())
            by.setdefault((d["cell"], d["params"], d["n_fitted"]), []).append(d)
        rows = []
        for (cell, params, nd), got in sorted(by.items()):
            pm = [{k: float(np.mean([g["per_model"][j][k] for g in got])) for k in METRICS}
                  for j in range(len(got[0]["per_model"]))]
            rows.append({"cell": cell, "params": params, "cap": got[0]["cap"],
                         "n_fitted": nd, "lr": got[0]["lr"],
                         "seeds": sorted(g["seed"] for g in got), "per_model": pm,
                         **{k: float(np.mean([m[k] for m in pm])) for k in METRICS}})
        rows += json.loads((SHARDS / "baseline.json").read_text())
        for f in sorted(SHARDS.glob("lr-*.json")):
            d = json.loads(f.read_text())
            lrs[f"{d['src']}|{d['params']}"] = d["lr"]
        git = lambda *a: subprocess.run(["git", *a], capture_output=True,
                                        text=True).stdout.strip()
        OUT.write_text(json.dumps({"provenance": {
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "protocol": "docs/PROTOCOL_SCALING.md", "data_seed": DATA_SEED, "steps": STEPS,
            "seeds": SEEDS, "fit_pool": FIT_POOL, "held": HELD, "lr": lrs},
            "rows": rows}, indent=2, default=float))
        bad = [f"{r['cell']} {r['params']} n={r['n_fitted']} seeds={r['seeds']}"
               for r in rows if r.get("seeds") and len(r["seeds"]) != SEEDS]
        print(f"merged {len(rows)} rows -> {OUT}")
        if bad:
            print("INCOMPLETE: " + "; ".join(bad))
        return 0

    raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
