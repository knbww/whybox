#!/usr/bin/env python
"""Fast preliminary read on the transfer question, off the cached bundles.

Trains one interpreter on the source cell at reduced budget and scores it
against the cheap baselines on every cell.  Not a substitute for a full run --
no ceilings, no output-only arm, fewer epochs -- but it answers "does anything
transfer to the LM at all" in minutes instead of hours.
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.harness import baseline_scores, evaluate, table
from mint.experiments.transfer import split_bundles
from mint.interpreter.baselines import BASELINES
from mint.interpreter.dataset import to_problem_set
from mint.interpreter.train import TrainConfig, predict, train_interpreter

CELLS = ["skirmish/mlp", "skirmish/tab_transformer", "clinic/mlp",
         "clinic/tab_transformer", "seqworld/transformer", "agreement/lm"]
SOURCE = "skirmish/mlp"


def load(cell: str, want: int = 12):
    d, f = cell.split("/")
    for path in sorted(glob.glob(f"results/cache/{d}-{f}-*.pt"), key=lambda p: -len(p)):
        b = torch.load(path, weights_only=False)
        if len(b) == want:
            return b
    return None


def main(epochs: int = 15) -> int:
    per_cell, all_b, offsets = {}, [], {}
    for c in CELLS:
        b = load(c)
        if b is None:
            print(f"  (skipping {c}: not cached yet)")
            continue
        per_cell[c] = b
        offsets[c] = len(all_b)
        all_b.extend(b)
    names = list(per_cell)
    ps = to_problem_set(all_b, domains=sorted({x.domain for x in all_b}), cells=names)
    fit, held = {}, {}
    for c in names:
        f_, h_ = split_bundles(per_cell[c], 4, seed=0)
        fit[c] = [i + offsets[c] for i in f_]
        held[c] = [i + offsets[c] for i in h_]

    bid = ps.bundle_id.numpy()
    rows_of = lambda ids: torch.as_tensor(np.nonzero(np.isin(bid, ids))[0])
    train_ps = ps.subset(rows_of(fit[SOURCE]))
    print(f"cells: {names}\nfitting on {SOURCE}: {len(train_ps)} problems, {epochs} epochs")

    t0 = time.time()
    a = train_interpreter(train_ps, TrainConfig(epochs=epochs, d_model=96, n_layers=3,
                                                batch_size=64),
                          {SOURCE.split("/")[0]: get_domain(SOURCE.split("/")[0]).spec.n_factors})
    print(f"trained in {time.time() - t0:.0f}s\n")

    rows = []
    for c in names:
        es = ps.subset(rows_of(held[c]))
        rows += evaluate(es, all_b, predict(a, es), "A_preview")
        for bn in ("first_order", "grad_x_act", "correlation", "random"):
            rows += evaluate(es, all_b, baseline_scores(es, BASELINES[bn]), bn)
        oracle = es.y_effect.numpy().copy()
        oracle[~es.mask.numpy()] = -1e9
        rows += evaluate(es, all_b, oracle, "oracle_effect")

    for c in names:
        sub = [r for r in rows if r["cell"] == c]
        tag = "source" if c == SOURCE else "TRANSFER"
        print(f"\n### {c}  [{tag}]")
        print(table(sub, ["mediation.spearman", "executed.sufficiency@3", "decoy_rejection"],
                    sort_by="executed.sufficiency@3"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 15))
