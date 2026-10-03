#!/usr/bin/env python
"""Two things, in this order.

**Corrected evaluation of the model that was already trained.** No retraining:
the generator is reconstructed from its seed and the cached packs, which is
deterministic, and verified to be so. It is then scored under the fixed metric --
sign and strength defined by the effect of the support a claim proposes, and the
instrument scored in the matched regime rather than allowed to execute what the
generator has to predict. This says what the existing system was doing all along
when measured properly, which has to be known before any change to the objective
can be credited.

**Then a model trained on the whole statement.** Its direction and magnitude
heads are conditioned on the kind and carrier it commits to, and the loss is
`clause losses + lambda * the same losses run free`, so the heads have to agree
with each other rather than each being right on its own states.

Primary: whole-statement exact, and executed validity. The clause accuracies are
diagnostic only.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal.structure import TYPES
from mint.domains import get_domain
from mint.generative import JointGenerator, formula_template, verify
from mint.generative.composite import Composite, bucket, support_effect
from mint.models import build_target
from scripts.composite_explore import ARCH, DOMAINS, build_one, generate, tensors, train

PRIMARY = ["composition", "verified_sign", "verified_strength"]
DIAGNOSTIC = ["type", "support_exact", "support_jaccard", "sign", "strength"]


def corrected_truth(pack, domain) -> Composite:
    """Kind and carrier as extracted; sign and strength from the carrier's own effect."""
    a = support_effect(pack["model"], domain, pack["raw"],
                       pack["support"].astype(bool)) / max(pack["logit_sd"], 1e-6)
    return Composite(pack["kind"], pack["support"].astype(bool), (a > 0).astype(int),
                     bucket(a), pack["live"])


def train_joint(tr, epochs=50, seed=0, lam=1.0, **kw):
    torch.manual_seed(seed)
    m = JointGenerator(**kw)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr["kind"])
    wk = torch.bincount(tr["kind"], minlength=len(TYPES)).float().clamp(min=1).pow(-0.5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            args = dict(unit_x=tr["unit_x"][b], unit_mask=tr["unit_mask"][b],
                        pos_x=tr["pos_x"][b], pos_mask=tr["pos_mask"][b],
                        global_x=tr["global_x"][b], cand=tr["cand"][b])
            forced = m(**args, force_kind=tr["kind"][b], force_carrier=tr["support"][b])
            free = m(**args)
            sup_loss = (F.binary_cross_entropy_with_logits(
                forced["support"], tr["support"][b], reduction="none")
                * tr["cand"][b]).sum() / tr["cand"][b].sum().clamp(min=1)
            clause = (F.cross_entropy(forced["kind"], tr["kind"][b], weight=wk) + sup_loss
                      + F.cross_entropy(forced["sign"], tr["sign"][b])
                      + F.cross_entropy(forced["strength"], tr["strength"][b]))
            # the joint term: right about direction and size given its own commitments
            joint = (F.cross_entropy(free["sign"], tr["sign"][b])
                     + F.cross_entropy(free["strength"], tr["strength"][b]))
            opt.zero_grad(); (clause + lam * joint).backward(); opt.step()
    return m.eval()


@torch.no_grad()
def generate_joint(m, pack, u, q, carrier: np.ndarray | None = None) -> Composite:
    """`carrier` forces the claim to be about a support the model did not choose.

    The hybrid arm depends on this. Reading the model's own conditional heads and
    then swapping a different carrier underneath them would produce a claim whose
    direction and magnitude were derived for a *different* support -- an
    incoherent statement, and a misleading arm.
    """
    t = tensors([pack], u, q)
    fc = None
    if carrier is not None:
        T = t["pos_x"].shape[1]
        c = np.zeros((int(pack["live"].sum()), T), dtype=np.float32)
        c[:, :carrier.shape[1]] = carrier[pack["live"]]
        fc = torch.as_tensor(c)
    o = m(t["unit_x"], t["unit_mask"], t["pos_x"], t["pos_mask"], t["global_x"], t["cand"],
          force_carrier=fc)
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


def scoreboard(title, rows):
    print(f"\n### {title}")
    print(f"{'method':22s}" + "".join(c.rjust(16) for c in PRIMARY)
          + "  |" + "".join(c.rjust(16) for c in DIAGNOSTIC))
    print("-" * (22 + 16 * len(PRIMARY) + 3 + 16 * len(DIAGNOSTIC)))
    for k, v in rows.items():
        v = [r for r in v if r]
        if not v:
            continue
        print(f"{k:22s}" + "".join(f"{np.mean([r[c] for r in v]):16.3f}" for c in PRIMARY)
              + "  |" + "".join(f"{np.mean([r[c] for r in v]):16.3f}" for c in DIAGNOSTIC))


def main(n_models: int = 12, n_focal: int = 80, seeds: int = 2, data_seed: int = 4) -> int:
    t0 = time.time()
    packs = {d: [build_one(d, s, n_focal, data_seed) for s in range(n_models)] for d in DOMAINS}
    u = max(p["unit_x"].shape[1] for d in DOMAINS for p in packs[d])
    q = max(p["pos_x"].shape[1] for d in DOMAINS for p in packs[d])
    half = n_models // 2
    print(f"packs ready in {time.time() - t0:.0f}s", flush=True)

    tr0 = tensors(packs[DOMAINS[0]][:half], u, q)
    a, b = train(tr0, seed=0), train(tr0, seed=0)
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                 b.state_dict().values()))
    print(f"reconstruction is deterministic: {same} "
          f"(checkpoints were not saved; the same seed and cached data rebuild them)")

    for src in DOMAINS:
        tgt = [x for x in DOMAINS if x != src][0]
        fit, held = packs[src][:half], packs[tgt][half:]
        tr = tensors(fit, u, q)
        maj = int(np.bincount(tr["kind"].numpy()).argmax())
        d = get_domain(tgt)
        rows = {k: [] for k in ("oracle", "instrument (matched regime)",
                                "instrument (allowed to execute)", "old model, corrected metric",
                                "joint model", "joint hybrid: measured carrier")}
        for sd in range(seeds):
            old = train(tr, seed=sd)
            new = train_joint(tr, seed=sd)
            for p in held:
                gtc = corrected_truth(p, d)
                args = (p["model"], d, p["raw"])
                g_old = generate(old, p, u, q)
                g_new = generate_joint(new, p, u, q)
                rows["old model, corrected metric"].append(
                    verify(*args, g_old, p["logit_sd"], gtc))
                rows["joint model"].append(verify(*args, g_new, p["logit_sd"], gtc))
                if sd == 0:
                    for name, ex in (("instrument (matched regime)", False),
                                     ("instrument (allowed to execute)", True)):
                        f = formula_template(p["model"], d, p["raw"], p["fo"], p["cand"],
                                             p["live"], p["logit_sd"], maj,
                                             execute_sign_strength=ex)
                        rows[name].append(verify(*args, f, p["logit_sd"], gtc))
                        if not ex:
                            g_h = generate_joint(new, p, u, q, carrier=f.support)
                            rows["joint hybrid: measured carrier"].append(verify(
                                *args, Composite(g_h.kind, f.support, g_h.sign,
                                                 g_h.strength, p["live"]),
                                p["logit_sd"], gtc))
                    rows["oracle"].append(verify(*args, gtc, p["logit_sd"], gtc))
        scoreboard(f"{src} -> {tgt}   (primary | diagnostic)", rows)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
