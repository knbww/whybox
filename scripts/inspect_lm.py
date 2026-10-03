#!/usr/bin/env python
"""Qualitative trace: what does the interpreter actually say about a sentence?

    python scripts/inspect_lm.py results/llm_transfer_arms.pt [arm] [n]

For each sampled state it prints the sentence, the human-annotated decisive
factor, how much the language model moves when that factor is neutralised in
the world, the units the interpreter proposes, and what actually happens when
those units are patched.  Numbers only mean something next to the sentence they
came from.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

from mint.causal import counterfactual_acts, subset_patch_effect
from mint.causal.layout import UnitLayout
from mint.domains import get_domain
from mint.interpreter.dataset import to_problem_set
from mint.interpreter.train import predict


def main(arms_path: str, arm: str = "A_full", n_show: int | str = 6) -> int:
    n_show = int(n_show)
    blob = torch.load(arms_path, weights_only=False)
    if arm not in blob["arms"]:
        print(f"available arms: {sorted(blob['arms'])}")
        return 1
    model_a = blob["arms"][arm]

    cache = sorted(glob.glob("results/cache/agreement-lm-*.pt"))
    if not cache:
        print("no cached agreement/lm bundles; run configs/llm_transfer.yaml first")
        return 1
    bundles = torch.load(cache[0], weights_only=False)
    held = [i - min(blob["held_ids"]["agreement/lm"]) for i in blob["held_ids"]["agreement/lm"]]
    b = bundles[held[0] % len(bundles)]

    domain = get_domain("agreement")
    ps = to_problem_set([b])
    scores = predict(model_a, ps)["mediation"]
    layout = UnitLayout.of(b.trained.model)
    x = torch.as_tensor(domain.to_model_input(b.focal_raw))
    cf = counterfactual_acts(b.trained.model, domain, b.focal_raw)
    dec = b.gt.decisive
    total = b.gt.factor_total[np.arange(len(dec)), dec]
    target_vals = cf[np.arange(len(dec)), :, dec]

    order = np.argsort(-scores, axis=1)
    m = np.zeros_like(scores, dtype=bool)
    rows = np.repeat(np.arange(len(dec))[:, None], 3, axis=1)
    m[rows, order[:, :3]] = True
    suff = subset_patch_effect(b.trained.model, layout, x, target_vals, m[:, :layout.n_units],
                               b.gt.base_logit)

    print(f"\narm: {arm}   target model: {b.tag}   units: {b.n_units}\n")
    picked = np.argsort(-np.abs(total))[:n_show]
    for i in picked:
        units = [layout.site_of(int(u)) for u in order[i, :3]]
        kinds = ["head" if s.startswith("attn") else "neuron" for s, _ in units]
        print(f"  {domain.to_sentence(b.focal_raw[i]):58s}")
        print(f"     human factor : {domain.spec.factors[dec[i]].name}")
        print(f"     LM response  : {total[i]:+.2f} logit when that factor is neutralised in the world")
        print(f"     A proposes   : " +
              ", ".join(f"{k} {s}#{j}" for (s, j), k in zip(units, kinds)))
        print(f"     patching them: {suff[i]:+.2f} logit "
              f"({suff[i] / total[i] * 100:.0f}% of the response reproduced)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
