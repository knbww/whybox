#!/usr/bin/env python
"""The diagnostic that motivated the volume study, made before its protocol.

Fitted `skirmish/mlp` models 0-11 only. The twelve held-out models are never
loaded into any computation here.

1. The confirmed interpreter (`follows_b.train_pointer`, 40 epochs, dropout 0.1,
   weight decay 1e-2), fitted on all twelve: its accuracy on its own training states.
2. The same architecture with no dropout and no weight decay, trained for up to 400
   epochs on nine fitted models, scored at epochs 40/100/200/400 on those nine and
   on the other three fitted models.

If (2) memorises its training states while accuracy on the three unseen fitted
models falls, the interpreter has the capacity to fit and lacks the data to
generalise.

    python -m scripts.volume_prereg_diagnostic     # ~6 min, writes results/volume_prereg_diagnostic.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mint.domains import get_domain
from scripts.follows_b import MARGIN, Pointer, load_population, prepare, recover_name, tensors, train_pointer

CHECKPOINTS = (40, 100, 200, 400)


def accuracy(m, tr, spec) -> float:
    m.eval()
    with torch.no_grad():
        return float((recover_name(m(*tr[:4]).numpy(), spec) == tr[5]).mean())


def unregularised(tr, va, spec, epochs: int = 400, seed: int = 0) -> list[dict]:
    torch.manual_seed(seed)
    ux, um, px, gx, carr, _ = tr
    m = Pointer(ux.shape[-1], px.shape[-1], gx.shape[-1], dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    out = []
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(carr), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(m(ux[b], um[b], px[b], gx[b]), carr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in CHECKPOINTS:
            out.append({"epoch": ep, "train_accuracy": accuracy(m, tr, spec),
                        "unseen_fitted_accuracy": accuracy(m, va, spec)})
            print(f"  epoch {ep:3d}: train {out[-1]['train_accuracy']:.4f}  "
                  f"unseen fitted {out[-1]['unseen_fitted_accuracy']:.4f}", flush=True)
    return out


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    d = get_domain("skirmish"); spec = d.spec
    bundles, path, sha = load_population("skirmish", 12)       # fitted models only
    assert {b.trained.cfg.seed for b in bundles} == {0, 1, 2, 3}
    fit = [prepare(b, d) for b in bundles]
    u_max = max(p["unit_x"].shape[1] for p in fit)

    tr = tensors(fit, u_max, spec, MARGIN)
    confirmed = accuracy(train_pointer(tr, spec, seed=0), tr, spec)
    print(f"confirmed recipe, train accuracy on 12 fitted models: {confirmed:.4f} "
          f"({len(tr[5])} states)", flush=True)
    curve = unregularised(tensors(fit[:9], u_max, spec, MARGIN), tensors(fit[9:], u_max, spec, MARGIN), spec)

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                          "cache": path, "cache_sha256": sha, "models": "fitted 0-11 only",
                          "margin": MARGIN, "interpreter_seed": 0, "torch": torch.__version__},
           "confirmed_recipe_train_accuracy": confirmed, "n_train_states": int(len(tr[5])),
           "unregularised_9_of_12": curve, "wall_seconds": time.time() - t0}
    with open("results/volume_prereg_diagnostic.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/volume_prereg_diagnostic.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
