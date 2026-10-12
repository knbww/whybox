# Pre-registration — the division of labour

Frozen before `hybrid_confirm` runs, on fresh target populations
(`data_seed: 5`). The exploratory run it follows is frozen as it stands and is
not recomputed.

## The claim, stated exactly

> A hybrid that takes the carrier from the analytic instrument and the kind,
> direction and magnitude from a learned interpreter conditioned on that carrier
> produces a more accurate whole causal statement than the aligned analytic
> instrument alone — **in both transfer directions**.

## Metrics

| metric | role |
|---|---|
| whole-statement exact accuracy (`composition`) | **primary** |
| executed validity — running the claim reproduces both its sign and its size | **co-primary** |
| carrier accuracy, type accuracy, sign accuracy, strength accuracy | diagnostic only |

Directions are reported and tested **separately** and never averaged: an
asymmetry between them is a result, not noise to be smoothed away.

## Arms, frozen

| arm | definition |
|---|---|
| `instrument_aligned` | greedy-by-first-order carrier; type named as the training majority, which is all a derivative can do; sign and strength estimated from the first-order sum over that carrier. **Matched regime**: it predicts, it does not execute. |
| `interpreter_joint` | `JointGenerator` over the frozen causal representation. Direction and magnitude heads conditioned on the kind and carrier the model commits to. Loss = clause losses + 1.0 × the same losses run free. |
| `hybrid` | carrier from `instrument_aligned`; kind, sign and strength from `interpreter_joint` **conditioned on that carrier**. |
| `random_composition` | random type, random non-empty subset of the candidates, random sign and strength. |
| `oracle` | the executed truth; a reference line, not a family member. |

The hybrid's conditioning is a correction made before this freeze: the
exploratory version read heads conditioned on the model's own carrier and then
substituted a different one, producing a statement whose direction and magnitude
were derived for a support it did not claim. Its exploratory numbers (0.087 and
0.082) are therefore **not comparable** to what this run produces.

The input representation is unchanged and will not be touched: the first-order
report is the instrument, and raw internals were already shown to add nothing.

## Design

| | |
|---|---|
| domains | `interact` and `witness` — same causal structures, disjoint surfaces |
| targets per domain | 24, **12 fitted / 12 held out** |
| interpreter seeds | 0, 1, 2 |
| data seed | 5 |
| pairing | over the same held-out target models, two-sided exact sign-flip |
| smallest attainable paired p | 0.00049 |

## Family, frozen — eight comparisons

| # | metric | comparison | direction |
|---|---|---|---|
| 1 | whole-statement exact | `hybrid` vs `instrument_aligned` | interact→witness |
| 2 | whole-statement exact | `hybrid` vs `instrument_aligned` | witness→interact |
| 3 | executed validity | `hybrid` vs `instrument_aligned` | interact→witness |
| 4 | executed validity | `hybrid` vs `instrument_aligned` | witness→interact |
| 5 | whole-statement exact | `hybrid` vs `interpreter_joint` | interact→witness |
| 6 | whole-statement exact | `hybrid` vs `interpreter_joint` | witness→interact |
| 7 | whole-statement exact | `hybrid` vs `random_composition` | interact→witness |
| 8 | whole-statement exact | `hybrid` vs `random_composition` | witness→interact |

Holm-Bonferroni over all eight, α = 0.05. Smallest attainable adjusted p is
0.0039, so the design can reach significance. Comparisons 1 and 2 are the ones
the claim stands or falls on; 5 and 6 establish that the division of labour is
doing the work rather than the learned model alone; 7 and 8 are the sanity floor.

No further comparisons will be added, and no diagnostic will be promoted to a
test after the fact.

## Gate

**CONFIRMED** — comparisons 1 and 2 both reject with a positive delta, and the
hybrid loses nothing in the family. The substantive conclusion may then be
written.

**ASYMMETRIC** — one of 1 or 2 rejects with a positive delta and the other does
not. The claim is then scoped to that direction, and the asymmetry is reported as
the finding it is rather than averaged away.

**FAIL** — neither 1 nor 2 rejects. The exploratory advantage does not replicate.

**VOID** — comparison 7 or 8 fails to reject while 1 or 2 does, which would mean
the family cannot tell the hybrid from a random claim and nothing in it can be
trusted.

## Not allowed after results

Adding features, adding comparisons, changing an arm, promoting a diagnostic,
averaging the directions, or recomputing the exploratory run.
