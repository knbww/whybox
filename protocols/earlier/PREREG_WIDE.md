# Pre-registration — the widened interface

Frozen before `scripts/wide_type.py` runs. Written because `docs/INTERFACE.md`
narrowed the interpreter's input to first-order marginals, and that narrowing has
a mechanical consequence: causal *type* is defined by how joint interventions
differ from the sum of single ones, so a marginal cannot express it. Type has sat
at the majority-class rate on every axis measured so far. This asks whether it is
absent from the model or only from the *view* of the model the interpreter was
given.

## The claim, stated exactly

> Causal type is legible from a target model's raw internal state and is not
> legible from a first-order summary of that state. A type reader given
> activations, gradients and executed single-unit effects reads the type of an
> unseen target model, in a domain whose surface it has never seen, more
> accurately than the identical reader given only the first-order causal
> pattern — **in both transfer directions**.

This is a test of the interpreting layer, not of the measurement layer. No arm
here competes with `first_order` at what `first_order` computes.

## Arms, frozen

| arm | unit-stream input |
|---|---|
| `narrow` | `causal_pattern` only — the frozen interface, `D_UNIT = 9` |
| `wide` | `causal_pattern` + `raw_state_pattern`, `D = 16` |
| `raw_only` | `raw_state_pattern` only, `D = 7` |
| `majority_of_source` | constant: the commonest type among the source-domain fitted models |
| `n_candidates_lookup` | parameter-free: the commonest type per candidate-count, fitted on source models only |

`raw_state_pattern` supplies activations, gradients, the executed single-unit
ablation effect, and the gap between that effect and its first-order prediction.
It excludes `first_order_cf` and `unit_ie`: both are indexed by factor and would
tell the reader which factor to name.

The two floors are a **sanity floor**, not the comparison the claim rests on.
They are how an artifact is told apart from an interpretation. `majority_of_source`
is a zero-information constant; `n_candidates_lookup` is the parameter-free rule
an audit found beats the learned reader in both directions.

All three learned arms are the identical architecture, capacity and training
budget. Only the visible feature block differs.

## Design

| | |
|---|---|
| domains | `interact` and `witness` — same causal structures, disjoint surfaces |
| targets per domain | 24, **12 fitted / 12 held out**, the same split the frozen runs use |
| reader seeds | 0, 1, 2 — averaged per held-out model before pairing |
| data seed | 5 |
| metric | balanced accuracy over the five causal types, per held-out model |
| pairing | over the same held-out target models, two-sided exact sign-flip |
| smallest attainable paired p | 0.000488 |

Balanced accuracy is used because the type distribution is uneven; under it a
constant predictor scores exactly 0.200, which makes the floor unambiguous.

## Family, frozen — eight comparisons

| # | comparison | direction |
|---|---|---|
| 1 | `wide` vs `narrow` | interact→witness |
| 2 | `wide` vs `narrow` | witness→interact |
| 3 | `raw_only` vs `narrow` | interact→witness |
| 4 | `raw_only` vs `narrow` | witness→interact |
| 5 | `wide` vs `majority_of_source` | interact→witness |
| 6 | `wide` vs `majority_of_source` | witness→interact |
| 7 | `wide` vs `n_candidates_lookup` | interact→witness |
| 8 | `wide` vs `n_candidates_lookup` | witness→interact |

Holm–Bonferroni over all eight, α = 0.05. Smallest attainable adjusted p is
0.0039, so the design can reach significance. **Comparisons 1 and 2 are the
primary and the claim stands or falls on them.** 3 and 4 say whether the
first-order block contributes anything once raw state is present. 5–8 are the
sanity floor.

## Gate

**CONFIRMED** — 1 and 2 both reject with a positive delta, and 5–8 all reject
with a positive delta. Type is legible from raw internal state, in both
directions, above every floor.

**ASYMMETRIC** — one of 1 or 2 rejects with a positive delta and the other does
not. The claim is scoped to that direction and the asymmetry is reported as the
finding it is, never averaged away.

**FAIL** — neither 1 nor 2 rejects. Raw internal state does not make type legible
at this scale, and the narrowing was not what was hiding it.

**VOID** — `wide` fails to beat `majority_of_source` (5 or 6 does not reject)
while 1 or 2 does. The setup cannot then tell an interpretation from a constant
and nothing in it may be reported as a positive result.

## Not allowed after results

Adding features, adding comparisons, changing an arm, promoting a diagnostic,
averaging the directions, or re-reading the gate on a metric other than balanced
accuracy of the type clause.
