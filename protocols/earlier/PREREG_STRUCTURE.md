# Pre-registration — does reading the target's internals add to transfer?

Frozen before `structure_confirm` runs, on fresh target populations
(`data_seed: 3`). Informed by the exploratory run recorded in `docs/RESULTS.md`,
which is what an exploratory run is for.

## The one question

The exploratory run showed the *kind* of causal structure transferring between
two surface-disjoint domains at 85–97% of the in-domain ceiling, well above an
analytic heuristic. It also showed that a reader seeing **only the per-position
first-order pattern** matched or beat the full model in three cells of four.

So the question that decides whether this project is about reading a model's
internal state, rather than about classifying a derivative better than a
hand-written rule, is exactly one:

> Does the internal-unit stream add to **transfer** beyond the position stream?

Everything else is already measured and is not re-litigated here.

## Design

| | |
|---|---|
| domains | `interact` and `witness` — same four causal structures, disjoint surfaces |
| targets per domain | 24, **12 fitted / 12 held out** |
| reader seeds | 0, 1, 2 |
| data seed | 3 (fresh populations) |
| metric | balanced accuracy over the five structure types, chance 0.200 |
| smallest attainable paired p | 0.00049 |
| declared comparisons | 4 × 2 transfer directions = 8 |
| smallest attainable adjusted p | 0.0039 — α = 0.05 is reachable |

Metrics are computed per held-out target model and averaged across reader seeds
before testing, so the bootstrap and test unit is the target model.

## Arms

`reader` (both streams) · `reader_positions_only` · `reader_units_only`

## Baselines, declared in full

`first_order_heuristic` — the same classification rules applied to the
first-order report instead of to execution. `majority_of_source` — the source
domain's most common type carried over, which catches frequency matching.
`random`. `in_domain_ceiling` — a reader fitted on the target domain.

## Declared comparisons

On each transfer direction (`interact → witness`, `witness → interact`):

1. `reader` vs `reader_positions_only` — **the primary one**
2. `reader` vs `first_order_heuristic`
3. `reader_positions_only` vs `first_order_heuristic`
4. `reader_units_only` vs `random`

Paired over held-out target models, two-sided exact sign-flip, Holm-Bonferroni
over the family of eight, α = 0.05.

## Gate

**Internals add** — comparison 1 rejects with a positive delta in at least one
direction and loses in neither. The claim becomes: reading the target's internal
state contributes to a transferable notion of causal structure.

**Measurement transfers, internals do not** — comparison 2 rejects but
comparison 1 does not. The claim is scoped to: a learned classification of a
universal causal measurement transfers across surface-disjoint domains and beats
hand-written rules over the same measurement. The internals stream is reported as
above chance (comparison 4) but not additive.

**Fail** — comparison 2 does not reject. The transfer seen exploratorily does not
replicate.

**Void** — `reader_units_only` fails to beat random (comparison 4) in both
directions while `reader` beats `positions_only`, which would indicate the gain
comes from something other than the internals and needs investigation before any
claim.

## Not allowed after results

Changing the metric, the arms, the family, or the gate. Any such change makes the
run exploratory and it must be labelled so.
