#!/usr/bin/env python
"""The interpreter states a whole causal claim about B, and every clause is executed.

Protocol frozen in `docs/PREREG_STATEMENT.md` before this ran.

The two runs before this one showed B's internal causal state is readable and
readable transferably, but in both the model's output was an index and the human
sentence was a lookup table written by hand. Here the model asserts all three
clauses -- which cause, which way B's output moves when it is removed, by how
much -- from B's internals with no probe, and each is checked by running the
intervention the model itself named.
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
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from mint.generative.composite import CUTS, SIGNS, STRENGTHS, bucket
from scripts.follows_b import (MARGIN, Pointer, carriers_in, load_population, prepare,
                               recover_name)

FAMILY = [("whole", "constant", "ALL"), ("whole", "random", "ALL"),
          ("cause", "constant", "ALL"), ("direction", "constant", "ALL"),
          ("magnitude", "constant", "ALL"), ("whole", "constant", "DISAGREE")]
DIRS = ("вниз", "вверх")
SIZES = ("слабо", "заметно", "сильно")


def say(spec, cause: int, sign: int, strength: int, scores=None) -> str:
    """The claim in words, assembled from what the model asserted and what the
    domain declares -- a factor's own name, its own description, and the names of
    the inputs that carry it. Nothing here is a phrasebook written by hand: drop
    in a different domain and the sentence comes out in that domain's words.
    """
    f = spec.factors[cause]
    _w = len(scores) if scores is not None else spec.n_inputs
    carried = ", ".join(spec.input_names[c] for c in carriers_in(spec, _w)[cause])
    out = [f"B опирается на: {carried}",
           f"  это {f.name} — {f.description}",
           f"  убрать → выход B пойдёт {DIRS[sign]}, {SIZES[strength]}"]
    if scores is not None:
        # What the model actually scored, per candidate, exactly as `recover_name`
        # reads it: the mean over that factor's carriers. Printing a bare top-k of
        # positions instead is misleading, because a single high position can lose
        # to a two-carrier factor whose other carrier is low.
        per = [(np.mean([scores[c] for c in cc]), g.name)
               for cc, g in zip(carriers_in(spec, _w), spec.factors)]
        rank = ", ".join(f"{n} {v:+.2f}" for v, n in sorted(per, reverse=True)[:3])
        out.append(f"  веса интерпретатора по причинам: {rank}")
    return "\n".join(out)


class Speaker(nn.Module):
    """Points at B's inputs *and* asserts the direction and size of the effect."""

    def __init__(self, d_unit, d_pos, d_global, d_model=96):
        super().__init__()
        self.trunk = Pointer(d_unit, d_pos, d_global, d_model)
        self.sign = nn.Sequential(nn.Linear(2 * d_model + d_global, d_model), nn.GELU(),
                                  nn.Linear(d_model, len(SIGNS)))
        self.strength = nn.Sequential(nn.Linear(2 * d_model + d_global, d_model), nn.GELU(),
                                      nn.Linear(d_model, len(STRENGTHS)))

    def forward(self, unit_x, unit_mask, pos_x, global_x):
        t = self.trunk
        hu = t.unit_enc(t.unit_in(unit_x), src_key_padding_mask=~unit_mask)
        u = (hu * unit_mask[..., None]).sum(1) / unit_mask.sum(1, keepdim=True).clamp(min=1)
        hp = t.pos_enc(t.pos_in(pos_x))
        pooled = torch.cat([u, hp.mean(1), global_x], -1)
        c = t.ctx(pooled)
        z = torch.cat([hp, c[:, None, :].expand(-1, hp.shape[1], -1)], -1)
        return {"pos": t.head(z).squeeze(-1), "sign": self.sign(pooled),
                "strength": self.strength(pooled)}


def _as_input(a) -> torch.Tensor:
    """A tabular target takes floats; a language target takes token ids. Forcing
    float32 here silently broke the language branch."""
    t = torch.as_tensor(a)
    return t.float() if t.is_floating_point() else t


def executed(model, domain, raw: np.ndarray, factor: np.ndarray, sd: float) -> np.ndarray:
    """(N,) B's actual shift when the *named* factor is neutralised, in B's own sd."""
    with torch.no_grad():
        base = model(_as_input(domain.to_model_input(raw)))[0].numpy()
    out = np.zeros(len(raw))
    for f in np.unique(factor):
        m = factor == f
        z = domain.to_model_input(domain.do_neutral(raw[m], int(f)))
        with torch.no_grad():
            out[m] = model(_as_input(z))[0].numpy() - base[m]
    return out / max(sd, 1e-9)


def truth(pack, domain, model) -> dict:
    """Labels: what B relies on, and what removing *that* actually does to B."""
    k = pack["margin"] > MARGIN
    raw, sd = pack["raw"][k], pack["sd"]
    eff = executed(model, domain, raw, pack["y"][k], sd)
    return {"cause": pack["y"][k], "sign": (eff > 0).astype(int),
            "strength": bucket(eff), "agree": pack["world"][k] == pack["y"][k],
            "raw": raw, "sd": sd, "keep": k}


def score(claim: dict, t: dict, domain, model) -> dict:
    """Every clause executed: direction and size are checked on the named cause."""
    eff = executed(model, domain, t["raw"], claim["cause"], t["sd"])
    ok_c = claim["cause"] == t["cause"]
    ok_s = claim["sign"] == (eff > 0).astype(int)
    ok_m = claim["strength"] == bucket(eff)
    return {"cause": ok_c, "direction": ok_s, "magnitude": ok_m,
            "whole": ok_c & ok_s & ok_m}


def tensors(packs, u_max, spec):
    keep = [p["margin"] > MARGIN for p in packs]
    pad = lambda a, n: np.pad(a, ((0, 0), (0, n - a.shape[1]), (0, 0)))
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    ux = np.concatenate([pad(p["unit_x"], u_max)[k] for p, k in zip(packs, keep)])
    um = np.concatenate([np.pad(np.ones((int(k.sum()), p["unit_x"].shape[1]), bool),
                                ((0, 0), (0, u_max - p["unit_x"].shape[1])))
                         for p, k in zip(packs, keep)])
    px = np.concatenate([p["pos_x"][k] for p, k in zip(packs, keep)])
    gx = np.concatenate([p["global_x"][k] for p, k in zip(packs, keep)])
    return T(ux), T(um, torch.bool), T(px), T(gx)


def train_speaker(packs, truths, u_max, spec, seed=0, epochs=40):
    torch.manual_seed(seed)
    ux, um, px, gx = tensors(packs, u_max, spec)
    y_c = np.concatenate([t["cause"] for t in truths])
    y_s = torch.as_tensor(np.concatenate([t["sign"] for t in truths]), dtype=torch.long)
    y_m = torch.as_tensor(np.concatenate([t["strength"] for t in truths]), dtype=torch.long)
    car = carriers_in(spec, px.shape[1])
    carr = np.zeros((len(y_c), px.shape[1]), np.float32)
    for i, f in enumerate(y_c):
        carr[i, car[f]] = 1.0
    carr = torch.as_tensor(carr)
    m = Speaker(ux.shape[-1], px.shape[-1], gx.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(y_c), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            o = m(ux[b], um[b], px[b], gx[b])
            loss = (F.binary_cross_entropy_with_logits(o["pos"], carr[b])
                    + F.cross_entropy(o["sign"], y_s[b])
                    + F.cross_entropy(o["strength"], y_m[b]))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def speak(m, pack, u_max, spec) -> dict:
    o = m(*tensors([pack], u_max, spec))
    return {"cause": recover_name(o["pos"].numpy(), spec),
            "sign": o["sign"].argmax(1).numpy(), "strength": o["strength"].argmax(1).numpy()}


def analytic(pack, spec, t, execute: bool, domain, model) -> dict:
    """Cause from the largest first-order sensitivity; sign and size from the
    first-order sum over the carriers it named -- unless allowed to execute."""
    k = pack["keep"] if "keep" in pack else pack["margin"] > MARGIN
    fo = pack["fo"][k]
    cause = recover_name(np.abs(fo), spec)
    car = carriers_in(spec, fo.shape[1])
    est = np.array([fo[i, car[c]].sum()
                    for i, c in enumerate(cause)]) / max(t["sd"], 1e-9)
    if execute:
        est = executed(model, domain, t["raw"], cause, t["sd"])
    return {"cause": cause, "sign": (est > 0).astype(int), "strength": bucket(est)}


def main(domain: str = "skirmish", family: str = "mlp", cache: str | None = None,
         out: str = "results/statement.json", n_models: int = 24, seeds: int = 3) -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    d = get_domain(domain); spec = d.spec
    n_models, seeds = int(n_models), int(seeds)   # CLI passes strings
    if cache:
        bundles, path, sha = torch.load(cache, weights_only=False)[:n_models], cache, None
    else:
        bundles, path, sha = load_population(domain, n_models)
    packs = [prepare(b, d) for b in bundles]
    for p, b in zip(packs, bundles):
        p["raw"], p["sd"] = b.focal_raw, float(b.gt.probe.logit_std)
    models_B = [b.trained.model for b in bundles]
    truths = [truth(p, d, m) for p, m in zip(packs, models_B)]
    u_max = max(p["unit_x"].shape[1] for p in packs)
    fit, held = range(12), range(12, 24)
    print(f"{domain}/{family}: обучение 12 моделей, проверка 12 невиданных "
          f"({time.time() - t0:.0f}s)", flush=True)

    speakers = []
    for sd_ in range(seeds):
        speakers.append(train_speaker([packs[i] for i in fit], [truths[i] for i in fit],
                                      u_max, spec, seed=sd_))
        print(f"    seed {sd_} обучен ({time.time() - t0:.0f}s)", flush=True)

    yc = np.concatenate([truths[i]["cause"] for i in fit])
    ys = np.concatenate([truths[i]["sign"] for i in fit])
    ym = np.concatenate([truths[i]["strength"] for i in fit])
    maj = (int(np.bincount(yc, minlength=spec.n_factors).argmax()),
           int(np.bincount(ys, minlength=2).argmax()),
           int(np.bincount(ym, minlength=3).argmax()))
    rng = np.random.default_rng(0)

    per: dict[tuple[str, str, str], list[float]] = {}
    for i in held:
        p, t, mB = packs[i], truths[i], models_B[i]
        n = len(t["cause"])
        claims = {
            "constant": {"cause": np.full(n, maj[0]), "sign": np.full(n, maj[1]),
                         "strength": np.full(n, maj[2])},
            "random": {"cause": rng.integers(0, spec.n_factors, n),
                       "sign": rng.integers(0, 2, n), "strength": rng.integers(0, 3, n)},
            "analytic": analytic(p | t, spec, t, False, d, mB),
            "analytic_executed": analytic(p | t, spec, t, True, d, mB),
        }
        sc = {nm: score(c, t, d, mB) for nm, c in claims.items()}
        ip = [score(speak(s, p, u_max, spec), t, d, mB) for s in speakers]
        sc["interpreter"] = {k: np.mean([x[k] for x in ip], axis=0) for k in ip[0]}
        for nm, s in sc.items():
            for metric in ("cause", "direction", "magnitude", "whole"):
                for sub, msk in (("ALL", np.ones(n, bool)), ("DISAGREE", ~t["agree"])):
                    if msk.sum():
                        per.setdefault((nm, metric, sub), []).append(
                            float(np.asarray(s[metric], float)[msk].mean()))

    print(f"\n{'арм':20s}{'cause':>9s}{'direction':>11s}{'magnitude':>11s}{'whole':>9s}"
          f"{'whole|DIS':>11s}")
    for nm in ("analytic_executed", "interpreter", "analytic", "constant", "random"):
        r = [np.mean(per[(nm, m, "ALL")]) for m in ("cause", "direction", "magnitude", "whole")]
        print(f"{nm:20s}{r[0]:>9.4f}{r[1]:>11.4f}{r[2]:>11.4f}{r[3]:>9.4f}"
              f"{np.mean(per[(nm, 'whole', 'DISAGREE')]):>11.4f}")

    stats, pv = [], []
    for metric, ref, sub in FAMILY:
        a = np.array(per[("interpreter", metric, sub)])
        b = np.array(per[(ref, metric, sub)])
        delta, p = paired_sign_flip(a, b)
        stats.append({"metric": metric, "method": "interpreter", "reference": ref,
                      "cell": sub, "n_models": len(a), "delta": delta,
                      "a": float(a.mean()), "b": float(b.mean())})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
    print(f"\n{'metric':12s}{'vs':12s}{'subset':11s}{'delta':>9s}{'p':>10s}{'p_adj':>10s}  sig")
    print("-" * 76)
    for r in stats:
        print(f"{r['metric']:12s}{r['reference']:12s}{r['cell']:11s}{r['delta']:>+9.4f}"
              f"{r['p']:>10.5f}{r['p_adj']:>10.5f}  {('+' if r['delta'] > 0 else '-') if r['reject'] else ''}")

    prim = stats[0]
    void = not stats[1]["reject"] and prim["reject"]
    clause = any(r["reject"] and r["delta"] > 0 for r in stats[2:5])
    gate = ("VOID" if void else
            "CONFIRMED" if prim["reject"] and prim["delta"] > 0 and stats[1]["reject"] else
            "PARTIAL" if clause else "FAIL")
    print(f"\nGATE: {gate}")

    # One stated claim, in words, assembled only from what the model asserted.
    p, t, mB = packs[12], truths[12], models_B[12]
    cl = speak(speakers[0], p, u_max, spec)
    j = int(np.argmax(t["agree"] == False)) if (~t["agree"]).any() else 0
    eff = executed(mB, d, t["raw"][j:j + 1], cl["cause"][j:j + 1], t["sd"])[0]
    with torch.no_grad():
        sc = speakers[0](*tensors([p], u_max, spec))["pos"].numpy()[j]
    print(f"\nОДНО ВЫСКАЗАННОЕ УТВЕРЖДЕНИЕ (модель #12, состояние {j}):")
    print("  " + say(spec, int(cl["cause"][j]), int(cl["sign"][j]),
                     int(cl["strength"][j]), sc).replace("\n", "\n  "))
    print(f"\n  исполнено на B: сдвиг {eff:+.3f} sd → на деле {DIRS[int(eff > 0)]}, "
          f"{SIZES[int(bucket(np.array([eff]))[0])]}")
    print(f"  на что B опирается на самом деле: "
          f"{spec.factors[int(t['cause'][j])].name}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    rows = [{"method": nm, "metric": m, "cell": s, "n_models": len(v),
             "per_model": [{"accuracy": x} for x in v], "accuracy": float(np.mean(v))}
            for (nm, m, s), v in per.items()]
    res = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "prereg": "docs/PREREG_STATEMENT.md", "domain": f"{domain}/{family}",
                          "cache": path, "cache_sha256": sha,
                          "margin": MARGIN, "cuts": CUTS, "n_models": n_models,
                          "seeds": list(range(seeds)), "torch": torch.__version__},
           "rows": rows, "comparisons": stats, "gate": gate,
           "wall_seconds": time.time() - t0}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
