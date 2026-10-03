#!/usr/bin/env python
"""Build the extra training volume for the volume study. Nothing here is scored.

Three populations on `skirmish/mlp`, none of which overlaps the twelve held-out
target models of the pinned population (models 12-23, seeds 4-7):

1. `fit_states` -- the twelve *fitted* models of the pinned population (seeds 0-3),
   measured again on 1280 fresh states each, with a fresh probe. The states axis
   uses nested prefixes of these states, so its rungs differ only in count.
2. `new_models` -- 84 new target models, seeds 8-35 x decoy strength 0/1/2, 160
   states each. The models axis adds them in seed order.
3. `val_models` -- 12 further models, seeds 36-39, used only to decide when to stop
   training. Never a training example, never a held-out target.

Seeds 4-7 are excluded from 2 and 3 on purpose: a target model's seed fixes its
initial weights *and* its architecture (`population` picks `archs[seed % 3]`), so
reusing a held-out seed would put a near-copy of a held-out model into training.

The recipe for new target models is the pinned population's own (6000 training
states, 60 epochs, weight decay 1e-4, no label smoothing), drawn from random
streams distinct from the pinned `data_seed 0`. File hashes go to
`results/volume_manifest.json`, which `volume_study.py` checks before it runs.

    python -m scripts.volume_build            # full
    python -m scripts.volume_build --smoke    # seconds; writes under results/cache/volume_smoke
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mint.causal.interventions import ground_truth
from mint.domains import get_domain
from mint.interpreter.dataset import Bundle, build_bundles
from mint.models.train_target import population
from scripts.follows_b import load_population

DOMAIN = "skirmish"
HELD_SEEDS = {4, 5, 6, 7}
FULL = dict(out=Path("results/cache/volume"), manifest=Path("results/volume_manifest.json"),
            n_fit=12, n_states=1280, n_probe=512, new_seeds=range(8, 36), val_seeds=range(36, 40),
            spurious=(0.0, 1.0, 2.0), n_focal=160, n_train=6000, epochs=60)
SMOKE = dict(out=Path("results/cache/volume_smoke"),
             manifest=Path("results/cache/volume_smoke/manifest.json"),
             n_fit=2, n_states=16, n_probe=64, new_seeds=range(8, 9), val_seeds=range(36, 37),
             spurious=(0.0, 1.0), n_focal=16, n_train=400, epochs=2)
STATES_STREAM, NEW_STREAM, VAL_STREAM = 101, 202, 303   # pinned population used 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_states(d, c) -> list[Bundle]:
    """The pinned fitted models, re-measured on fresh states from their own decoy strength."""
    pinned, _, _ = load_population(DOMAIN, 12)
    seeds = {b.trained.cfg.seed for b in pinned}
    assert seeds == {0, 1, 2, 3}, f"fitted seeds changed: {sorted(seeds)}"
    out = []
    for i, b in enumerate(pinned[:c["n_fit"]]):
        t0 = time.time()
        rng = np.random.default_rng(STATES_STREAM * 1000 + i)
        sp = b.trained.cfg.spurious_strength
        focal = d.sample(c["n_states"], rng, spurious_strength=sp).raw
        probe = d.sample(c["n_probe"], rng, spurious_strength=sp).raw
        gt = ground_truth(b.trained.model, d, focal, probe)
        out.append(Bundle(b.trained, gt, None, focal, DOMAIN))
        print(f"  fit_states [{i + 1}/{c['n_fit']}] {b.trained.cfg.tag():34s} "
              f"{len(focal)} states ({time.time() - t0:.1f}s)", flush=True)
    return out


def new_population(d, c, seeds, stream) -> list[Bundle]:
    assert not HELD_SEEDS & set(seeds), "a held-out seed would leak a near-copy into training"
    cfgs = population(d, seeds=tuple(seeds), spurious=c["spurious"], epochs=c["epochs"],
                      n_train=c["n_train"], weight_decay=1e-4, label_smoothing=0.0, family="mlp")
    return build_bundles(DOMAIN, cfgs, n_focal=c["n_focal"], n_probe=c["n_probe"], seed=stream)


def main(smoke: bool = False) -> int:
    torch.set_num_threads(4)
    c = SMOKE if smoke else FULL
    c["out"].mkdir(parents=True, exist_ok=True)
    d = get_domain(DOMAIN)
    t0 = time.time()
    parts = {"fit_states": lambda: fit_states(d, c),
             "new_models": lambda: new_population(d, c, c["new_seeds"], NEW_STREAM),
             "val_models": lambda: new_population(d, c, c["val_seeds"], VAL_STREAM)}
    manifest = {"domain": DOMAIN, "smoke": smoke, "streams": {
        "fit_states": STATES_STREAM, "new_models": NEW_STREAM, "val_models": VAL_STREAM},
        "config": {k: (list(v) if isinstance(v, range) else str(v) if isinstance(v, Path) else v)
                   for k, v in c.items()}, "files": {}}
    for name, build in parts.items():
        print(f"{name} ({time.time() - t0:.0f}s)", flush=True)
        bundles = build()
        path = c["out"] / f"{name}.pt"
        torch.save(bundles, path)
        manifest["files"][name] = {"path": str(path), "sha256": sha256(path),
                                   "n_models": len(bundles),
                                   "seeds": sorted({b.trained.cfg.seed for b in bundles})}
    c["manifest"].write_text(json.dumps(manifest, indent=1))
    print(f"wrote {c['manifest']} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(smoke="--smoke" in sys.argv[1:]))
