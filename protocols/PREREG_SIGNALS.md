# Pre-registration — does the interpreter read B better from dynamic signals?

Frozen before `scripts/signals_study.py` runs. Follows `docs/PREREG_VOLUME.md`.

> **Authorship note.** This protocol text was drafted with an AI assistant (Claude)
> after the author chose the direction ("check dynamic signals — gradients and the
> like") on 2026-09-15, and committed by the author's instruction before the run.
> It is an internal protocol of this repository, not the competition research plan.

## Why this run exists

`results/volume_study.json`: eight times more training volume, of either kind, moved
accuracy from 0.7432 to 0.7500 / 0.7501 (no detectable change in any of four
comparisons), and at the chosen checkpoint the interpreter sat at train ~0.77,
validation ~0.73 on every rung. More examples do not lift it. The remaining
explanation inside the project's own formulation is that the signals it reads do
not carry enough about the cause.

Everything the interpreter reads today is a **snapshot at the state**: a gradient
times a displacement, at `x`. The target models are ReLU networks, so a
second-derivative term at `x` is zero almost everywhere; what the snapshot misses
lies **along the path** from the state to the reference.

## The claim, stated exactly

> Given dynamic signals of B measured along the counterfactual path, the same
> interpreter names the cause B relies on more accurately than from the snapshot it
> reads today — and the gain is not merely the effect of giving it more inputs.

## Signals

`mint.encoding.dynamics.path_signals`, on the path from `x` to the reference `r`
(every declared factor neutral at once — the reference the snapshot already uses),
with 16 segments:

| where | signal | definition |
|---|---|---|
| input position | integrated gradient | (r − x) × mean over the path of ∂f/∂x; sums to f(r) − f(x) |
| input position | gradient variability | \|r − x\| × std over the path of ∂f/∂x |
| internal unit | conductance | Σ over segments of ∂f/∂a at the midpoint × Δa across the segment; sums to f(r) − f(x) within each layer |
| internal unit | activation response | a(r) − a(x), over the unit's spread on the probe set |
| state | total change | f(r) − f(x), over B's logit spread on the probe set |

The two completeness identities and the zero at positions no factor carries are
tested in `tests/test_dynamics.py`.

**What this costs the design, stated plainly.** Measuring the path runs B on 17
interpolated inputs forward and 16 backward. Nothing is factor-indexed: the path is
the same all-factors-neutral path for every state, and B is never run with one
factor removed. But the dynamic arms no longer satisfy "the interpreter reads B
without running it on any other input" (`PREREG_STATEMENT.md`). A result from them
must say so.

## Arms — one interpreter, only the inputs differ

| arm | unit features | position features | global |
|---|---|---|---|
| `current` | the confirmed 9 | the confirmed 5 | the confirmed 8 |
| `+snapshot` | + activation z-score and gradient at `x`, separately (10) | 5 | 8 |
| `+dynamics` | + conductance, activation response (10) | + integrated gradient, gradient variability (10) | + total change, log IG mass (2) |
| `+both` | + both (20) | 10 more | 2 more |

Each new per-unit or per-position quantity is expanded exactly as the existing ones
(signed and absolute over the state's maximum, tied rank, share, sign). Activation
z-scores, normalised gradients and activation responses are clipped to ±10.
`+snapshot` is the control: more inputs about the same moment.

## Fixed

| | |
|---|---|
| domain | `skirmish/mlp`, seven causes |
| training | the 12 pinned fitted models (seeds 0–3), their pinned 160 states |
| stopping | 12 validation models, seeds 36–39 (`results/volume_manifest.json`); up to 200 epochs, best validation checkpoint, patience 20 — as `PREREG_VOLUME.md` |
| held-out | the 12 pinned models 12–23 and their states, as `results/follows_b.json` |
| label | `argmax abs(gt.factor_total)`, margin > 0.10 |
| interpreter | `follows_b.Pointer`, d_model 96, 2 layers; AdamW lr 1e-3, weight decay 1e-2, batch 64 |
| interpreter seeds | 0, 1, 2, averaged per held-out model before pairing |
| metric | cause-name accuracy per held-out model, ALL and DISAGREE reported separately |

## Family, frozen — six comparisons

| # | arm | vs | subset |
|---|---|---|---|
| 1 | `+snapshot` | `current` | ALL |
| 2 | `+snapshot` | `current` | DISAGREE |
| 3 | `+dynamics` | `current` | ALL |
| 4 | `+dynamics` | `current` | DISAGREE |
| 5 | `+both` | `current` | ALL |
| 6 | `+both` | `current` | DISAGREE |

Two-sided exact sign-flip, paired over the 12 held-out models; Holm over these six,
α = 0.05. Smallest attainable p 0.000488, adjusted 0.0029.

## Read-out

Each comparison is read on its own: **better** if it rejects with a positive delta,
**worse** if negative, **no detectable change** otherwise. Nothing is pooled.

The claim is supported if comparison 3 or 5 reads **better**. Whether the gain is
specific to dynamics is read from the diagnostic below together with comparisons
1–2; it is not a separate gate.

**DISAGREE is the guard against a world-predictor.** Richer activations make it
easier to learn the simulator from B's internal state; a system that did that would
gain on AGREE and lose on DISAGREE. Any gain on ALL is read next to DISAGREE.

**Diagnostic**, outside the family, unadjusted: `+dynamics` vs `+snapshot` and
`+both` vs `+dynamics`, on ALL and DISAGREE.

**Floors**, outside the family, reported in every table: `world_oracle`,
`first_order_pointer`, `ig_pointer` (the same parameter-free carrier rule applied to
integrated gradients), `constant`, `random`. `ig_pointer` is a measurement; how it
compares with the interpreter is not what this run asks.

## Disclosed before the run

While debugging the smoke test (2 models, 4 path segments, 2 training epochs), the
script printed its table on **two held-out models (12, 13)** before this was noticed:
the floors there read `ig_pointer` 0.695 and `first_order_pointer` 0.595 on ALL; the
arm numbers came from 2 epochs on 2 models and carry no information. The smoke test
was then changed to use fitted models as stand-ins. No full-size number exists for
any arm or floor.

## Qualification that travels with every number

The twelve held-out models are four independent seeds at three decoy strengths
(ERRATA §11); p-values assume twelve units and must be quoted with this sentence.

## Not allowed after results

Changing the signals, their normalisation, the path length, the arms, the stopping
rule or the family; pooling subsets; promoting a diagnostic or a floor; re-running
with other interpreter seeds and reporting the preferred run.
