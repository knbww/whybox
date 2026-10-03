#!/usr/bin/env python
"""Exploratory probe of the semantic layer. No gate, no confirmatory claim.

    python scripts/semantic_explore.py

Asks only whether the pipeline works at all and whether anything leaks:
  agreement/lm -> held-out agreement/lm   (does naming generalise across models)
  agreement/lm -> polarity/lm             (does it cross to another legible task)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from mint.experiments.transfer import ExperimentConfig, build_or_load, split_bundles
from mint.semantic import (SemanticConfig, baselines, build, leakage_checks, predict,
                           score, train)

SOURCE, TARGET = "agreement/lm", "polarity/lm"


def main(seeds: int = 2, epochs: int = 25) -> int:
    cfg = ExperimentConfig(seeds=(0, 1, 2, 3, 4, 5, 6, 7), spurious=(0.0, 1.0, 2.0),
                           target_n_train=6000, target_epochs=60, n_focal=160,
                           n_probe=512, data_seed=1, holdout_models=12)
    pops = {c: build_or_load(c.split("/")[0], c.split("/")[1], cfg) for c in (SOURCE, TARGET)}
    fit_b, held_b = {}, {}
    for c, bs in pops.items():
        f, h = split_bundles(bs, cfg.holdout_models, seed=cfg.data_seed)
        fit_b[c], held_b[c] = [bs[i] for i in f], [bs[i] for i in h]

    train_ds = build(fit_b[SOURCE], cells=[SOURCE])
    evals = {c: build(held_b[c], cells=[c]) for c in (SOURCE, TARGET)}
    ceil_ds = build(fit_b[TARGET], cells=[TARGET])
    print(f"train {len(train_ds)} problems from {SOURCE}; "
          + ", ".join(f"eval {c}:{len(v)}" for c, v in evals.items()))
    for m in leakage_checks(train_ds, fit_b[SOURCE]):
        print("  leakage check:", m)

    arms = {
        "A_semantic": dict(use_locus=True, blind=False),
        "A_no_locus": dict(use_locus=False, blind=False),
        "A_blind": dict(use_locus=True, blind=True),
    }
    rows = []
    for name, over in arms.items():
        t0 = time.time()
        preds = {c: [] for c in evals}
        for sd in range(seeds):
            m = train(train_ds, SemanticConfig(epochs=epochs, seed=sd, **over),
                      verbose=(name == "A_semantic" and sd == 0))
            for c, es in evals.items():
                preds[c].append(predict(m, es).argmax(1))
        for c, es in evals.items():
            best = max(range(seeds), key=lambda i: 0)  # seed-average below
            per_seed = [score(es, p, name)[0] for p in preds[c]]
            merged = dict(per_seed[0])
            for key in ("balanced_accuracy", "accuracy"):
                merged[key] = float(np.mean([r[key] for r in per_seed]))
                merged[key + ".seed_sd"] = float(np.std([r[key] for r in per_seed]))
            rows.append(merged)
        print(f"  {name} done in {time.time() - t0:.0f}s")

    print("\n== in-cell ceiling on the transfer cell ==")
    for sd in range(seeds):
        m = train(ceil_ds, SemanticConfig(epochs=epochs, seed=sd), verbose=False)
        rows.append(score(evals[TARGET], predict(m, evals[TARGET]).argmax(1),
                          f"ceiling_{TARGET} (seed {sd})")[0])

    for c, es in evals.items():
        for bname, pred in baselines(es).items():
            rows.append(score(es, pred, bname)[0])
        rows.append(score(es, es.y.numpy(), "oracle")[0])

    for c in evals:
        sub = sorted([r for r in rows if r["cell"] == c],
                     key=lambda r: -r["balanced_accuracy"])
        tag = "SOURCE, held-out models" if c == SOURCE else "TRANSFER"
        print(f"\n### {c}  [{tag}]   ({sub[0]['n_models']} models, "
              f"{int(sub[0]['n_candidates'])} candidates)")
        print(f"{'method':32s}{'balanced acc':>18s}{'accuracy':>18s}")
        print("-" * 68)
        for r in sub:
            sd = r.get("balanced_accuracy.seed_sd")
            extra = f" (seed sd {sd:.3f})" if sd else ""
            print(f"{r['method']:32s}{r['balanced_accuracy']:12.3f}"
                  f"±{r['balanced_accuracy.ci']:.3f}{r['accuracy']:12.3f}"
                  f"±{r['accuracy.ci']:.3f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
