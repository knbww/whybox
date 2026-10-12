# Pre-registration — does the reading transfer to a different ontology?

Frozen before `scripts/transfer_b.py` runs. Follows `docs/PREREG_FOLLOWS_B.md`,
whose gate returned CONFIRMED in-cell on `skirmish/mlp`.

## Why this run exists

Transfer is not the novelty; it is the check. If the interpreter learnt a
principle for reading a model, it keeps working on target models whose ontology,
cause vocabulary, cause count and carrier layout it has never seen. If it merely
memorised one environment, it does not.

## The claim, stated exactly

> An interpreter fitted on target models of one domain, reading only B's internal
> ML signals, names the cause of B's decision on target models of a *different*
> domain — including on the states where those models rely on something other
> than what drives their world.

## Design

| | |
|---|---|
| domains | `skirmish/mlp` (7 causes, chance 0.143) and `clinic/mlp` (5 causes, chance 0.200) |
| directions | `skirmish→clinic` and `clinic→skirmish`, **reported separately, never averaged** |
| targets | 24 per domain; fitted on the source's first 12, evaluated on the target's last 12 |
| label | `argmax abs(gt.factor_total)` of the *target* model, margin > 0.10 (as Amendment 1 of the previous protocol) |
| interpreter seeds | 0, 1, 2, averaged per held-out model before pairing |
| pairing | over the same 12 held-out models, two-sided exact sign-flip |
| smallest attainable paired p | 0.000488 |

Nothing about the target domain reaches the interpreter. It emits one score per
input position; the name is recovered afterwards using the *target* domain's
carrier table, which is a reporting step, never an input. The two domains share
neither causes, nor their number, nor their carrier layout: `skirmish` pairs
positions 0–1 for MAN_ADVANTAGE, `clinic` pairs 1–2 for HYPOXIA and 0–7 for
FRAILTY.

The position stream carries no absolute index and the encoder has no positional
encoding, so the interpreter cannot transfer by memorising a fixed slot.

## Arms

| arm | what it is |
|---|---|
| `interpreter` | fitted on the source domain only |
| `world_oracle` | always names the target world's cause; 1.000 on AGREE, **0.000 on DISAGREE** |
| `first_order_pointer` | parameter-free, reads B; domain-agnostic by construction, so it transfers trivially and is the floor to clear |
| `constant_source` | the commonest cause *index* among the source's fitted models, applied blindly in the target. Indices mean different things in the two domains, which is the point: a transferred constant carries no target knowledge |
| `random` | uniform over the target domain's causes |

## Family, frozen — six comparisons

| # | comparison | subset | direction |
|---|---|---|---|
| 1 | `interpreter` vs `constant_source` | **DISAGREE** | skirmish→clinic |
| 2 | `interpreter` vs `constant_source` | **DISAGREE** | clinic→skirmish |
| 3 | `interpreter` vs `random` | DISAGREE | skirmish→clinic |
| 4 | `interpreter` vs `random` | DISAGREE | clinic→skirmish |
| 5 | `interpreter` vs `constant_source` | ALL | skirmish→clinic |
| 6 | `interpreter` vs `constant_source` | ALL | clinic→skirmish |

Holm–Bonferroni over all six, α = 0.05; smallest attainable adjusted p 0.0029.
**Comparisons 1 and 2 are the primary.**

`first_order_pointer` and `world_oracle` appear in every table as the floor and
the ceiling-of-the-wrong-kind respectively. They are not in the family: the
analytic pointer is expected to transfer, and beating it is not what this run is
for.

## Gate

**CONFIRMED** — 1 and 2 both reject with a positive delta. The reading transfers
in both directions.

**ASYMMETRIC** — one of 1 or 2 rejects with a positive delta and the other does
not. Scoped to that direction and reported as the asymmetry it is.

**FAIL** — neither rejects.

**VOID** — 3 or 4 fails to reject while its paired primary rejects, meaning the
subset cannot separate the interpreter from a coin.

## Not allowed after results

Adding arms, adding comparisons, averaging the two directions, pooling AGREE and
DISAGREE, or changing the margin.
