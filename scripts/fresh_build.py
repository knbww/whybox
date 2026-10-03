#!/usr/bin/env python
"""Build twelve initialisations that nothing in this project has ever scored.

Why. Every initialisation up to 39 has been used for something: 0..7 are the pinned
population, 8..35 were the volume study's training pool, 36..39 its early-stopping
set, and 24..35 were all scored by `scripts/seeds12.py` and by its audit. A
confirmatory run needs a held-out set nobody has looked at. This builds one.

What. Initialisations 40..51, each at decoy strengths 0, 1 and 2: 36 targets, with
the exact recipe of `scripts/volume_build.py::new_population` (6000 training
states, 60 epochs, weight decay 1e-4, no label smoothing, 160 focal and 512 probe
states), from random stream 404, which no other build uses (0, 101, 202, 303 are
taken). The architecture follows the initialisation (`archs[seed % 3]`), so the
twelve are 4 / 4 / 4 across the three architectures.

Nothing is scored here. The builder prints each target's own test AUC, which is a
property of B on its own task and says nothing about the interpreter. No label,
no disagreement count and no accuracy of any arm is computed or printed.

    python -m scripts.fresh_build     # writes results/cache/fresh/held.pt and
                                      # results/fresh_manifest.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from mint.domains import get_domain
from scripts.volume_build import DOMAIN, FULL, new_population, sha256

SEEDS = range(40, 52)
STREAM = 404
OUT = Path("results/cache/fresh/held.pt")
MANIFEST = Path("results/fresh_manifest.json")


def main() -> int:
    torch.set_num_threads(4)
    if OUT.exists() or MANIFEST.exists():
        print(f"{OUT} or {MANIFEST} already exists; the fresh population is built once.")
        return 2
    t0 = time.time()
    d = get_domain(DOMAIN)
    bundles = new_population(d, FULL, SEEDS, STREAM)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundles, OUT)
    targets = [{"initialisation": b.trained.cfg.seed,
                "strength": float(b.trained.cfg.spurious_strength),
                "widths": list(b.trained.cfg.arch.get("widths", ()))} for b in bundles]
    manifest = {"domain": DOMAIN, "stream": STREAM, "seeds": list(SEEDS),
                "strengths": list(FULL["spurious"]),
                "recipe": {k: FULL[k] for k in ("n_focal", "n_probe", "n_train", "epochs")},
                "file": {"path": str(OUT), "sha256": sha256(OUT), "n_models": len(bundles)},
                "targets": targets,
                "scored_before_confirmatory_run": False}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({len(bundles)} targets) and {MANIFEST} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
