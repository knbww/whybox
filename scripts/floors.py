#!/usr/bin/env python
"""Every non-learned floor, in one place, on the pinned held-out populations.

The paper quotes a first-order pointer at 0.887 on `skirmish`, but the arm stored
in `results/follows_b.json` aggregates |attribution| by the mean over a cause's
carriers and scores 0.626.  Both are the same instrument read by different
aggregation rules, and until now the stronger rule lived only in a scratch
script, so no number in the paper's baseline row traced to committed code.

This computes all of them, from the same pinned caches and behind the same
margin filter every arm is scored behind:

  fo_mean_abs      mean |first-order term| over the cause's carriers
  fo_sum_abs       sum of |first-order term|
  fo_abs_signed    |signed sum| -- the linearisation of do(f := neutral), which
                   is what the label is measured by, so it is the strongest of
                   the three wherever a cause has more than one carrier
  ig_abs_signed    the same rule on integrated gradients along the path to the
                   all-neutral reference (tabular targets only)

and three floors that are not read off B at all:

  state_displacement  how far the cause's carriers sit from their neutral value,
                      in units of the input's own spread.  Reads the state, never
                      the model.  This is the "maybe the cause is just whichever
                      feature looks unusual" control.
  live_random         uniform choice among the causes that are actually in play
                      at this state, i.e. whose do(f) is not a literal no-op.
                      Reported as its exact expectation rather than a draw.  On a
                      language target most causes are absent from most sentences,
                      so this floor is high and the nominal 1/K chance level is
                      not the difficulty of the task.
  constant / random / world_oracle, as in the confirmatory runs.

Writes `results/floors.json`.  No interpreter is trained here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.encoding.dynamics import path_signals
from mint.generative.task import _contrast

from follows_b import MARGIN, carriers_in, load_population, pos_first_order

DOMAINS = ("skirmish", "clinic", "entail")
N_MODELS = 24
N_FIT = 12


def labels(gt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(label, world's factor, margin) -- exactly as `follows_b.prepare` builds them."""
    a = np.abs(gt.factor_total) / max(gt.probe.logit_std, 1e-9)
    s = np.sort(a, axis=1)
    return a.argmax(1), gt.decisive, s[:, -1] - s[:, -2]


def by_carrier(v: np.ndarray, car: list[list[int]], how: str) -> np.ndarray:
    """(N, T) per-position quantity -> (N,) named cause, under one aggregation rule."""
    if how == "mean_abs":
        per = np.stack([np.abs(v)[:, c].mean(1) for c in car], 1)
    elif how == "sum_abs":
        per = np.stack([np.abs(v)[:, c].sum(1) for c in car], 1)
    elif how == "abs_signed":
        per = np.stack([np.abs(v[:, c].sum(1)) for c in car], 1)
    else:
        raise KeyError(how)
    return per.argmax(1)


def arms_for(bundle, domain, spec, maj: int) -> tuple[dict[str, np.ndarray], dict]:
    """Predictions of every floor at every state of one target model."""
    gt = bundle.gt
    y, world, margin = labels(gt)
    fo = pos_first_order(bundle.trained.model, domain, bundle.focal_raw)
    car = carriers_in(spec, fo.shape[1])
    n = len(y)

    out = {
        "fo_mean_abs": by_carrier(fo, car, "mean_abs"),
        "fo_sum_abs": by_carrier(fo, car, "sum_abs"),
        "fo_abs_signed": by_carrier(fo, car, "abs_signed"),
        "constant": np.full(n, maj),
        "world_oracle": world,
    }
    if spec.kind == "tabular":
        ig = path_signals(bundle.trained.model, domain, bundle.focal_raw,
                          gt.probe.act_std, gt.probe.logit_std)["ig"]
        out["ig_abs_signed"] = by_carrier(ig, car, "abs_signed")

        x = np.asarray(domain.to_model_input(bundle.focal_raw), dtype=np.float64)
        r = np.asarray(domain.to_model_input(_contrast(domain, bundle.focal_raw)), dtype=np.float64)
        z = np.abs(x - r) / (x.std(0) + 1e-9)
        out["state_displacement"] = np.stack([z[:, c].sum(1) for c in car], 1).argmax(1)

    # `live_random` and `random` are scored by their exact expectation, not a draw:
    # a sampled arm adds noise that has nothing to do with the floor's strength.
    live = np.abs(gt.factor_total) > 1e-9
    expect = {
        "live_random": np.where(live[np.arange(n), y], 1.0 / np.maximum(live.sum(1), 1), 0.0),
        "random": np.full(n, 1.0 / spec.n_factors),
    }
    return out, expect


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.time()
    rows, prov_cache = [], {}

    for name in DOMAINS:
        domain = get_domain(name)
        spec = domain.spec
        bundles, path, sha = load_population(name, N_MODELS)
        prov_cache[name] = {"path": path, "sha256": sha}

        fit_labels = []
        for b in bundles[:N_FIT]:
            y, _, m = labels(b.gt)
            fit_labels.append(y[m > MARGIN])
        maj = int(np.bincount(np.concatenate(fit_labels), minlength=spec.n_factors).argmax())

        per: dict[tuple[str, str], list[float]] = {}
        counts = {"ALL": 0, "AGREE": 0, "DISAGREE": 0}
        for b in bundles[N_FIT:]:
            y, world, margin = labels(b.gt)
            k = margin > MARGIN
            agree = world == y
            hard, soft = arms_for(b, domain, spec, maj)
            scored = {m: (p == y).astype(float) for m, p in hard.items()} | soft
            for sub, msk in (("ALL", k), ("AGREE", k & agree), ("DISAGREE", k & ~agree)):
                counts[sub] += int(msk.sum())
                for m, ok in scored.items():
                    per.setdefault((m, sub), []).append(float(ok[msk].mean()) if msk.sum() else np.nan)

        print(f"{name}: {counts['ALL']} оцениваемых состояний на {N_MODELS - N_FIT} отложенных "
              f"моделях, из них {counts['DISAGREE']} расходящихся ({time.time() - t0:.0f}s)",
              flush=True)
        for (m, sub), vals in sorted(per.items()):
            v = [x for x in vals if np.isfinite(x)]
            rows.append({"domain": name, "method": m, "cell": sub, "n_models": len(v),
                         "n_states": counts[sub], "per_model": [{"accuracy": x} for x in v],
                         "accuracy": float(np.mean(v))})
            print(f"     {m:20s}{sub:10s}{np.mean(v):.4f}")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "caches": prov_cache, "margin": MARGIN, "n_models": N_MODELS,
                          "n_fit": N_FIT, "torch": torch.__version__},
           "rows": rows, "wall_seconds": time.time() - t0}
    with open("results/floors.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/floors.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
