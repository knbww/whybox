# Pre-registration — does naming the cause improve with training volume?

Frozen before `scripts/volume_study.py` runs. Follows `docs/PREREG_FOLLOWS_B.md`,
whose gate returned CONFIRMED.

> **Authorship note.** This protocol text was drafted with an AI assistant (Claude)
> from design decisions the author confirmed on 2026-09-15, and committed by the
> author's instruction before the run. It is an internal protocol of this
> repository, not the competition research plan.

## Why this run exists

The confirmed interpreter names the cause the target model relies on at 0.727 on
held-out models. A diagnostic made **before this protocol**, on the twelve fitted
models only (the held-out models were not used), asks what limits it:

| | on its training states | on fitted models it was not trained on |
|---|---|---|
| confirmed recipe, fitted on all 12 | 0.7603 | — |
| no dropout, no weight decay, 9 of 12 — epoch 40 | 0.7404 | 0.7303 |
| epoch 100 | 0.8881 | 0.6616 |
| epoch 200 | 0.9991 | 0.6361 |
| epoch 400 | 0.9928 | 0.6438 |

Source: `scripts/volume_prereg_diagnostic.py` → `results/volume_prereg_diagnostic.json`.

The interpreter can memorise its training states, and the longer it does, the
worse it reads models it has not seen. That is the signature of too little
training data, not of too few parameters.

## The claim, stated exactly

> At a fixed interpreter — the confirmed `follows_b.Pointer`, 0.515M parameters,
> unchanged hyperparameters — accuracy in naming the cause the target model relies
> on rises with the volume of training, without adding causes or worlds.

## Two axes of volume, matched in total states

| rung | total states | **states axis** | **models axis** |
|---|---|---|---|
| 0 | 1 920 | 12 fitted models × 160 states | (the same cell) |
| 1 | 3 840 | 12 × 320 | 24 × 160 |
| 2 | 7 680 | 12 × 640 | 48 × 160 |
| 3 | 15 360 | 12 × 1280 | 96 × 160 |

- **States axis.** The twelve fitted models of the pinned population (seeds 0–3),
  measured again on 1280 fresh states each with a fresh probe. Rungs are nested
  prefixes, so they differ only in count.
- **Models axis.** The same twelve models at 160 states, plus new target models
  added in seed order: seeds 8–35 × decoy strength 0/1/2, 160 states each, trained
  with the pinned population's recipe (6000 states, 60 epochs, weight decay 1e-4).
- **Validation.** Twelve further models, seeds 36–39, used only to choose when to
  stop training. Never trained on, never scored.
- **Seeds 4–7 are excluded** from every training and validation population: a target
  model's seed fixes its initial weights and its architecture, and 4–7 are the
  held-out models.

All three populations are pinned by sha256 in `results/volume_manifest.json`, built
by `scripts/volume_build.py`. Nothing in that build is scored.

## What does not change

| | |
|---|---|
| domain | `skirmish/mlp`, the same seven causes |
| held-out | the same 12 pinned models (12–23) **and the same states** as `results/follows_b.json` |
| label | `argmax abs(gt.factor_total)`, margin > 0.10 |
| interpreter | `follows_b.Pointer`, d_model 96, 2 layers, dropout 0.1; AdamW lr 1e-3, weight decay 1e-2, batch 64 |
| interpreter seeds | 0, 1, 2, averaged per held-out model before pairing |
| metric | accuracy of the recovered cause name, per held-out model, on ALL states and on DISAGREE states (B's cause ≠ the world's), reported separately |

## Training length

Chosen per cell on the validation models: train up to **200 epochs**, keep the
checkpoint with the best validation accuracy, stop after **20 epochs** without
improvement. A fixed epoch count would be unfair to the larger rungs, because the
diagnostic above shows epoch count alone moves accuracy on unseen models by 0.09.

Rung 0 therefore differs from the confirmed run in its states and its stopping
rule. Its accuracy is reported next to 0.727, not tested against it.

## Family, frozen — four comparisons

| # | axis | subset | comparison |
|---|---|---|---|
| 1 | states | ALL | rung 3 vs rung 0 |
| 2 | states | DISAGREE | rung 3 vs rung 0 |
| 3 | models | ALL | rung 3 vs rung 0 |
| 4 | models | DISAGREE | rung 3 vs rung 0 |

Two-sided exact sign-flip test, paired over the 12 held-out models; Holm over these
four, α = 0.05. Smallest attainable p 0.000488, adjusted 0.0020.

## Read-out

Each comparison is read on its own: **rises** if it rejects with a positive delta,
**falls** if it rejects with a negative delta, **no detectable change** otherwise.
The answer for an axis and a subset is that comparison's reading. Subsets and axes
are never pooled into one verdict.

**Secondary**, outside the family, unadjusted: the slope of accuracy per doubling of
volume, per held-out model, tested against zero, for each axis and subset.

**Diagnostic**, outside the family, unadjusted: models axis against states axis at
rung 3, ALL and DISAGREE — whether diversity of models or number of states is worth
more at equal volume. It is not promoted after the run.

## Qualification that travels with every number

The twelve held-out models are four independent seeds at three decoy strengths
(ERRATA §11). Every comparison here is paired within those same models, so the
direction of an effect is meaningful; the p-values assume twelve units and must be
quoted with this sentence.

## Not allowed after results

Changing the rungs, the stopping rule, the validation models, the margin or the
family; pooling subsets or axes; promoting the secondary or diagnostic analyses;
re-running with different interpreter seeds and reporting the preferred run.
