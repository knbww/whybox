#!/usr/bin/env python
"""The interpreter, fitted on this project's toy language models, read on GPT-2.

Diagnostic. No protocol was frozen before this ran and nothing here is a claim.

Every target model measured so far was built inside this project. This takes the
interpreter trained on the twelve fitted `entail` targets -- two-layer
transformers of width 48, about forty thousand parameters each -- and applies it,
without a single gradient step of adaptation, to a pretrained GPT-2 solving a
different task in a different template with a different cause vocabulary. Three
thousand times the parameters, weights nobody here trained.

Two arms, differing only in which signal of the target the interpreter reads:

  snapshot   the first-order term at the sentence, which is what every committed
             result in this project is built on
  ig         the same quantity integrated along the path from the sentence to its
             all-neutral reference

On the toy targets the two are close. On GPT-2 they are not, and the reason is
measurable rather than rhetorical: a token substitution is a large move, and the
gradient at one end of it does not describe it. The path integral, summed over
positions, reproduces the measured change in the target's output; the snapshot
does not.

The per-unit and global streams are zeroed in both arms, for training and for
evaluation alike, because `results/ablation_streams.json` finds them inert -- and
because GPT-2 has tens of thousands of units, which the unit encoder could not
read at any reasonable cost. What is transferred is a reading of B's input
positions.

Writes `results/external_gpt2.json`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from mint.domains import get_domain
from mint.eval.stats import paired_sign_flip
from mint.semantic.encoding import D_GLOBAL, D_UNIT

import external_lm
from external_lm import (Target, all_neutral, factors, ground_truth, sample,
                         sim_lm_path_signals)
from follows_b import MARGIN, load_population, pos_features, pos_first_order, prepare

SOURCE, N_MODELS, N_FIT = "entail", 24, 12
SEEDS, STEPS = 3, 16
ARMS = ("snapshot", "ig", "multi")
N_FOCAL, N_PROBE, DATA_SEED = 160, 256, 0
MODEL_NAME = "gpt2"


def blank_streams(n: int, u: int = 1) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n, u, D_UNIT), np.float32), np.zeros((n, D_GLOBAL), np.float32)


def stack_features(parts: dict[str, np.ndarray], signal: str) -> np.ndarray:
    """(N, T, 5 * k): the five within-state transforms of each requested signal.

    With one signal and one carrier per cause, naming reduces to the argmax of a
    monotone function of a single number, and there is nothing a learned reader
    can add to the rule that takes it. Several signals per position is the
    cheapest way to make the map from signals to cause multivariate.
    """
    want = {"snapshot": ("snapshot",), "ig": ("ig",),
            "multi": ("ig", "gvar", "snapshot")}[signal]
    return np.concatenate([pos_features(parts[w]) for w in want], axis=-1)


def source_packs(signal: str) -> list[dict]:
    """The fitted and held-out toy targets, read through one signal set."""
    d = get_domain(SOURCE)
    bundles, _, _ = load_population(SOURCE, N_MODELS)
    out = []
    for b in bundles:
        p = prepare(b, d)
        parts = {"snapshot": pos_first_order(b.trained.model, d, b.focal_raw)}
        if signal != "snapshot":
            parts |= sim_lm_path_signals(b.trained.model, d, b.focal_raw, STEPS)
        ux, gx = blank_streams(len(p["y"]))
        out.append({"unit_x": ux, "global_x": gx, "pos_x": stack_features(parts, signal),
                    "y": p["y"], "world": p["world"], "margin": p["margin"],
                    "fo": parts.get("ig", parts["snapshot"])})
    return out


def target_pack(signal: str, target: Target, focal, probe,
                neutral: str = "fixed", refs: int = 4) -> dict:
    """The external model read the same way: labels by executing do(f) on it,
    features from its own input positions.

    With data-drawn neutrals there is no single reference sentence to integrate
    towards, so the path signals are averaged over `refs` draws of it. The label
    is averaged over its own draws inside `ground_truth`.
    """
    import external_lm as _E
    gt = ground_truth(target, focal, probe, neutral=neutral)
    ids = target.ids(focal)
    if neutral == "fixed":
        references = [target.ids([all_neutral(s) for s in focal])]
    else:
        rng = np.random.default_rng(1000)
        references = [target.ids([_E.resample_all(s, rng) for s in focal])
                      for _ in range(refs)]
    parts = {"snapshot": np.mean([target.position_first_order(ids, r)
                                  for r in references], 0)}
    if signal != "snapshot":
        got = [target.position_path_signals(ids, r, STEPS) for r in references]
        parts |= {k: np.mean([g[k] for g in got], 0) for k in got[0]}
    ux, gx = blank_streams(len(focal))
    return {"unit_x": ux, "global_x": gx, "pos_x": stack_features(parts, signal),
            "fo": parts.get("ig", parts["snapshot"]),
            "y": gt["y"], "world": gt["world"], "margin": gt["margin"],
            "world_group": gt["world_group"], "world_single": gt["world_single"],
            "logit_std": gt["logit_std"], "factor_total": gt["factor_total"]}


def as_tensors(pack: dict, u_max: int, keep: np.ndarray):
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    ux = np.pad(pack["unit_x"], ((0, 0), (0, u_max - pack["unit_x"].shape[1]), (0, 0)))[keep]
    um = np.zeros((int(keep.sum()), u_max), bool)
    um[:, :pack["unit_x"].shape[1]] = True
    return T(ux), T(um, torch.bool), T(pack["pos_x"][keep]), T(pack["global_x"][keep])


def train(packs: list[dict], carriers: list[list[int]], u_max: int, seed: int, epochs: int = 40):
    import torch.nn.functional as F
    from follows_b import Pointer
    torch.manual_seed(seed)
    parts = [as_tensors(p, u_max, p["margin"] > MARGIN) for p in packs]
    ys = np.concatenate([p["y"][p["margin"] > MARGIN] for p in packs])
    ux = torch.cat([p[0] for p in parts]); um = torch.cat([p[1] for p in parts])
    px = torch.cat([p[2] for p in parts]); gx = torch.cat([p[3] for p in parts])
    carr = torch.zeros(len(ys), px.shape[1])
    for i, f in enumerate(ys):
        carr[i, carriers[f]] = 1.0
    m = Pointer(ux.shape[-1], px.shape[-1], gx.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(carr), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(m(ux[b], um[b], px[b], gx[b]), carr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def name(m, pack: dict, carriers: list[list[int]], u_max: int, keep: np.ndarray) -> np.ndarray:
    s = m(*as_tensors(pack, u_max, keep)).numpy()
    return np.stack([s[:, c].mean(1) for c in carriers], 1).argmax(1)


def cells(pred: np.ndarray, y: np.ndarray, agree: np.ndarray) -> dict[str, float]:
    ok = (pred == y).astype(float)
    return {"ALL": float(ok.mean()),
            "AGREE": float(ok[agree].mean()) if agree.any() else float("nan"),
            "DISAGREE": float(ok[~agree].mean()) if (~agree).any() else float("nan")}


# Cells on the external target. The grammar has a decisive cause only where exactly
# one grammatical cause is present, so "B departs from the grammar" is defined only
# there. The other two groups are reported on their own rather than folded in:
#   NONE  no grammatical cause: whatever B relies on, the grammar names nothing
#   OVER  both present: either suffices, removing one barely moves the verb, and the
#         largest single-cause effect is often an attractor for that reason alone
EXT_CELLS = ("ALL", "SINGLE_AGREE", "SINGLE_DISAGREE", "NONE", "OVER")


def ext_masks(pack: dict, keep: np.ndarray) -> dict[str, np.ndarray]:
    g, w, y = pack["world_group"][keep], pack["world_single"][keep], pack["y"][keep]
    single = g == "single"
    return {"ALL": np.ones(len(y), bool),
            "SINGLE_AGREE": single & (w == y), "SINGLE_DISAGREE": single & (w != y),
            "NONE": g == "none", "OVER": g == "over"}


def ext_cells(ok: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    ok = np.asarray(ok, float)
    return {c: float(ok[m].mean()) if m.any() else float("nan") for c, m in masks.items()}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out", default="results/external_gpt2.json")
    ap.add_argument("--template", default="single", choices=list(external_lm.TEMPLATES))
    ap.add_argument("--neutral", default="fixed", choices=("fixed", "data"))
    a_ = ap.parse_args(argv)
    model_name, out_path = a_.model, a_.out
    external_lm.TEMPLATE = a_.template
    torch.set_num_threads(4)
    t0 = time.time()
    src_spec = get_domain(SOURCE).spec
    src_car = [list(f.carriers) for f in src_spec.factors]
    tgt_car = [list(c) for _, c, _ in factors()]

    rng = np.random.default_rng(DATA_SEED)
    focal, probe = sample(N_FOCAL, rng), sample(N_PROBE, rng)
    target = Target(model_name)
    print(f"{model_name}: {target.n_params / 1e6:.1f}M параметров ({time.time() - t0:.0f}s)",
          flush=True)

    rows, per_model, ext_per_seed, packs = [], {}, {}, {}
    for signal in ARMS:
        src = source_packs(signal)
        u_max = max(p["unit_x"].shape[1] for p in src)
        tgt = target_pack(signal, target, focal, probe, neutral=a_.neutral)
        packs[signal] = tgt
        print(f"[{signal}] сигналы посчитаны ({time.time() - t0:.0f}s)", flush=True)

        k_t = tgt["margin"] > MARGIN
        masks = ext_masks(tgt, k_t)
        models = []
        for s in range(SEEDS):
            models.append(train(src[:N_FIT], src_car, u_max, s))
            print(f"    seed {s} обучен ({time.time() - t0:.0f}s)", flush=True)

        held = []
        for p in src[N_FIT:]:
            k = p["margin"] > MARGIN
            y, agree = p["y"][k], p["world"][k] == p["y"][k]
            acc = np.mean([(name(m, p, src_car, u_max, k) == y) for m in models], 0)
            held.append({sub: float(acc[msk].mean()) if msk.sum() else np.nan for sub, msk in
                         (("ALL", np.ones_like(agree)), ("AGREE", agree), ("DISAGREE", ~agree))})
        per_model[signal] = held
        for sub in ("ALL", "AGREE", "DISAGREE"):
            v = [h[sub] for h in held if np.isfinite(h[sub])]
            rows.append({"arm": signal, "target": f"{SOURCE} (12 отложенных)", "cell": sub,
                         "n_models": len(v), "accuracy": float(np.mean(v)),
                         "per_model": [{"accuracy": x} for x in v]})

        y_t = tgt["y"][k_t]
        seeds_acc = [ext_cells(name(m, tgt, tgt_car, u_max, k_t) == y_t, masks) for m in models]
        ext_per_seed[signal] = seeds_acc
        for c in EXT_CELLS:
            rows.append({"arm": signal, "target": model_name, "cell": c, "n_models": 1,
                         "n_states": int(masks[c].sum()),
                         "accuracy": float(np.mean([a[c] for a in seeds_acc])),
                         "per_seed": [a[c] for a in seeds_acc]})
        print(f"[{signal}] {SOURCE} отложенные ALL {np.nanmean([h['ALL'] for h in held]):.4f}"
              f"   |   {model_name}: " + "  ".join(
                  f"{c} {np.mean([a[c] for a in seeds_acc]):.4f}" for c in EXT_CELLS), flush=True)

    # Floors, from the ig pack whatever the arm order, on the same kept states.
    tgt = packs["ig"]
    k_t = tgt["margin"] > MARGIN
    masks = ext_masks(tgt, k_t)
    y_t = tgt["y"][k_t]
    K = len(factors())
    live = (np.abs(tgt["factor_total"]) > 1e-9)[k_t]
    ig_est = np.abs(np.stack([tgt["fo"][:, list(c)].sum(1) for _, c, _ in factors()], 1))[k_t]
    floors = {
        "random": {c: 1.0 / K for c in EXT_CELLS},
        "live_random": ext_cells(np.where(live[np.arange(len(y_t)), y_t],
                                          1.0 / np.maximum(live.sum(1), 1), 0.0), masks),
        "ig_pointer": ext_cells(ig_est.argmax(1) == y_t, masks),
        # Defined only where the grammar names a cause; NaN elsewhere by definition.
        "grammar_oracle": {c: (float((tgt["world_single"][k_t][m] == y_t[m]).mean())
                               if m.any() and c.startswith("SINGLE") else float("nan"))
                           for c, m in masks.items()},
    }
    # Best single constant per cell, chosen with the answers in hand: an upper
    # envelope for any constant, not a baseline anyone could have fixed in advance.
    best = {}
    for c, m in masks.items():
        best[c] = (float(max(np.mean(y_t[m] == f) for f in range(K))) if m.any()
                   else float("nan"))
    floors["best_constant_envelope"] = best
    for nm, d_ in floors.items():
        for c, v in d_.items():
            rows.append({"arm": nm, "target": model_name, "cell": c, "n_models": 0,
                         "n_states": int(masks[c].sum()), "accuracy": v})

    a = np.array([h["ALL"] for h in per_model["ig"]])
    b = np.array([h["ALL"] for h in per_model["snapshot"]])
    delta, p = paired_sign_flip(a, b)

    print(f"\nсостояний: " + ", ".join(f"{c} {int(m.sum())}" for c, m in masks.items()))
    print(f"{'арм':24s}" + "".join(f"{c:>17s}" for c in EXT_CELLS))
    print("-" * (24 + 17 * len(EXT_CELLS)))
    for signal in ARMS:
        g = ext_per_seed[signal]
        print(f"{signal:24s}" + "".join(f"{np.mean([x[c] for x in g]):>17.4f}" for c in EXT_CELLS))
    for nm, d_ in floors.items():
        print(f"{nm:24s}" + "".join(f"{d_[c]:>17.4f}" for c in EXT_CELLS))
    print(f"\nig против snapshot на отложенных {SOURCE}: {delta:+.4f} (p {p:.4f}; 12 моделей = "
          f"4 независимые инициализации, на уровне инициализаций пол p = 0.125)")

    git = lambda *a_: subprocess.run(["git", *a_], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "role": "diagnostic, no protocol frozen before the run",
                          "source": SOURCE, "external_model": model_name, "template": a_.template,
                          "neutral": a_.neutral,
                          "external_params": int(target.n_params),
                          "n_focal": N_FOCAL, "n_probe": N_PROBE, "data_seed": DATA_SEED,
                          "steps": STEPS, "margin": MARGIN,
                          "interpreter_seeds": list(range(SEEDS)),
                          "streams": "unit and global zeroed in both arms",
                          "external_cells": list(EXT_CELLS),
                          "grammar_label": "defined only where exactly one grammatical cause "
                                           "is present (external_lm.grammar_group)",
                          "held_out_note": "entail models 12-23 are 4 independent "
                                           "initialisations x 3 decoy strengths",
                          "torch": torch.__version__},
           "rows": rows,
           "comparisons": [{"metric": "accuracy", "method": "ig", "reference": "snapshot",
                            "cell": "ALL", "target": SOURCE, "n_models": len(a),
                            "delta": delta, "a": float(a.mean()), "b": float(b.mean()),
                            "p": p}],
           "wall_seconds": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
