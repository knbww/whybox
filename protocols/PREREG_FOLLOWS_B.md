# Pre-registration — does the interpreter follow the model, or the world?

Frozen before `scripts/follows_b.py` runs.

## Why this run exists

Every measurement in this project up to 2026-09-13 trained and scored on
`gt.decisive`, which `domains/base.py::annotate` computes from `domain.logit` —
the **world's** decisive factor. It does not depend on the target model at all.
A system fitted on it is a predictor of the simulator that uses B only as a
window onto the state. That is not interpretation.

The model-level quantity was always there: `gt.factor_total[n, f]` is B's own
behavioural response to `do(f := neutral)`, obtained by running B. The two
disagree on **18.6%** of `skirmish` states, because the domain carries a decoy
that correlates with the outcome and causes nothing, and B sometimes learns it.

## The claim, stated exactly

> On the states where the target model relies on something other than what drives
> the world, an interpreter reading only B's internal ML signals names **B's**
> cause — not the world's. No predictor of the world can do this, by
> construction.

## Design

| | |
|---|---|
| domain | `skirmish/mlp` — the plainest vocabulary available: `allies_alive`, `bomb_planted`, `time_left` |
| targets | 24, **12 fitted / 12 held out**, `data_seed 0` |
| label | `argmax abs(gt.factor_total)` — the factor whose removal moves **B** most |
| interpreter seeds | 0, 1, 2, averaged per held-out model before pairing |
| metric | accuracy of the recovered cause name, per held-out model |
| pairing | over the same 12 held-out models, two-sided exact sign-flip |
| smallest attainable paired p | 0.000488 |

**The states are split and reported separately, never pooled into one headline:**

- **AGREE** — B's decisive factor equals the world's.
- **DISAGREE** — it does not. *This is where the question lives.*

## No menu of causes

The interpreter emits **one score per input position of B** and the cause name is
recovered afterwards by asking whose carriers those positions are. It is never
handed a list of candidate causes. Recovery is lossless: with a perfect pointer
it returns the right name on 1.0000 of states in both `skirmish` and `clinic`,
and no single carrier position resolves to the wrong factor.

What the interpreter sees: per-unit descriptors of B's internal state, per-input-
position first-order sensitivity of B, and architecture-normalised global
context. It does not see B's raw inputs, B's feature names, the human factor
vocabulary, the candidate mask, or any factor-indexed measurement.

## Arms

| arm | what it is |
|---|---|
| `interpreter` | the learned pointer described above |
| `world_oracle` | always names the world's cause. A *perfect* predictor of the simulator. Scores 1.000 on AGREE and **0.000 on DISAGREE by construction** |
| `first_order_pointer` | parameter-free: point at the input position with the largest first-order sensitivity of B, recover the name |
| `constant` | the commonest model-level cause among the fitted models |
| `random` | uniform over the domain's factors |

`first_order_pointer` and `constant` are the **sanity floor** — they are how an
artifact is told apart from an interpretation. They are not the comparison the
claim rests on, and this run is not a contest between a learned model and a
formula. Note that `first_order_pointer` also reads B, so it is *expected* to
pass the DISAGREE test; only `world_oracle` is expected to fail it, and that is
the point being demonstrated.

## Family, frozen — six comparisons

| # | comparison | subset |
|---|---|---|
| 1 | `interpreter` vs `constant` | **DISAGREE** |
| 2 | `interpreter` vs `world_oracle` | **DISAGREE** |
| 3 | `interpreter` vs `random` | DISAGREE |
| 4 | `interpreter` vs `constant` | AGREE |
| 5 | `interpreter` vs `random` | AGREE |
| 6 | `interpreter` vs `constant` | ALL |

Holm–Bonferroni over all six, α = 0.05. Smallest attainable adjusted p is
0.0029, so the design can reach significance. **Comparisons 1 and 2 are the
primary and the claim stands or falls on them.**

## Gate

**CONFIRMED** — 1 and 2 both reject with a positive delta. The interpreter names
B's cause where B departs from the world, above the constant floor and above a
perfect predictor of the world.

**FAIL** — 1 or 2 does not reject.

**VOID** — 3 fails to reject while 1 or 2 rejects, which would mean the DISAGREE
subset cannot separate the interpreter from a coin and nothing in it can be
trusted.

## Not allowed after results

Adding arms, adding comparisons, pooling AGREE and DISAGREE into one number,
changing the label back to `gt.decisive`, or promoting a diagnostic.

---

## Amendment 1 — the label needs a margin, declared before the run

Written after inspecting **label statistics only**; no arm has been trained and
no outcome is known. The counts below are properties of the target population,
not results.

`argmax abs(gt.factor_total)` is only a meaningful statement about what B relies
on when B actually separates its top factor from the runner-up. It often does
not, and precisely on the states this run is about:

| subset | median margin between 1st and 2nd factor, in units of B's own logit sd |
|---|---|
| AGREE | 0.352 |
| DISAGREE | **0.093** |

So most raw disagreements are near-ties, where the argmax is close to a coin
flip and no system could be asked to reproduce it.

**All scoring is therefore restricted to states where that margin exceeds 0.10.**
The threshold is chosen for statistical power, from this table:

| margin | states kept | disagreements | per model |
|---|---|---|---|
| 0.00 | 3840 | 715 | 29.8 |
| **0.10** | **2984** | **338** | **14.1** |
| 0.15 | 2619 | 225 | 9.4 |
| 0.25 | 2057 | 120 | 5.0 |

0.25 leaves five disagreement states per model, too thin to pair over. 0.10
keeps fourteen.

The same comparisons will additionally be reported at 0.15 and 0.25 as a
**sensitivity strip**. That strip sits outside the Holm family of six and is not
part of the gate; it is published whatever it shows, so that the choice of 0.10
can be judged.
