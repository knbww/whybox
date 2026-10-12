# Pre-registration — the interpreter states the whole claim, and every clause is executed

Frozen before `scripts/statement.py` runs.

## Why this run exists

The two confirmatory runs before this one established that B's internal causal
state is readable, and readable transferably. Neither establishes the thing the
project is actually for: that the interpreter **emits an interpretation**. In both
of them the model's output was an index, and the human sentence was a lookup
table written by hand. "Emit a precise, human-legible causal interpretation" was
never tested, only "guess the right label".

## The claim, stated exactly

> Reading only B's internal ML signals, the interpreter states a complete causal
> claim about B — *which cause, which way B's output moves when it is removed,
> and by how much* — and the claim survives being executed on B, more often than
> a claim assembled without reading B.

The sentence handed to a person is assembled **only from clauses the model
asserted**:

> «Причина — потеря численного преимущества. Убрать её — и оценка победы
> вырастет заметно (примерно на 0.13).»

## What the interpreter asserts

| clause | how it is produced | how it is checked |
|---|---|---|
| `cause` | scores over B's input positions; the name is recovered from whose carriers they are | against `argmax abs(gt.factor_total)` — the factor whose removal actually moves B most |
| `direction` | asserted, without probing | execute `do_neutral` for the cause **the model named** and read the sign |
| `magnitude` | asserted as WEAK / MODERATE / STRONG, without probing | same execution, bucketed at 0.75 and 1.75 of B's own logit spread |
| `whole` | the conjunction | all three right at once |

Direction and magnitude are scored against the effect of the cause *the model
itself named*, not of the true cause: the claim says "remove this and that
happens", so that is what gets run. Naming the wrong cause fails the `cause`
clause and therefore `whole`, but its own direction and magnitude can still be
checked, and are.

**The model never probes.** It asserts direction and magnitude from B's internals
alone. An arm permitted to execute its own proposal is trivially right about both
and is reported separately as a reference line, never inside the family — that
error cost this project a whole headline once already.

## Design

| | |
|---|---|
| domain | `skirmish/mlp`, in-cell: fitted on 12 targets, evaluated on 12 never read |
| label | `argmax abs(gt.factor_total)`, margin > 0.10 (as `PREREG_FOLLOWS_B.md` Amendment 1) |
| seeds | 0, 1, 2, averaged per held-out model before pairing |
| pairing | over the same 12 held-out models, two-sided exact sign-flip |
| smallest attainable adjusted p | 6 × 2/2¹² = 0.0029 |

## Arms

| arm | what it asserts |
|---|---|
| `interpreter` | all three clauses, from B's internals, no probe |
| `constant` | the commonest cause, direction and bucket among the fitted models |
| `random` | uniform cause, direction and bucket |
| `analytic` | cause from the largest first-order sensitivity; direction and magnitude from the first-order sum over the named carriers. **Matched regime: it predicts, it does not execute** |
| `analytic_executed` | same cause, but direction and magnitude read off by running the patch. Scores 1.000 on both by construction. A **reference line, outside the family** |

`analytic` and `analytic_executed` are the sanity floor and the ceiling-of-the-
wrong-kind. Neither is in the family and the claim does not rest on either.

## Family, frozen — six comparisons

| # | metric | comparison |
|---|---|---|
| 1 | `whole` | `interpreter` vs `constant` |
| 2 | `whole` | `interpreter` vs `random` |
| 3 | `cause` | `interpreter` vs `constant` |
| 4 | `direction` | `interpreter` vs `constant` |
| 5 | `magnitude` | `interpreter` vs `constant` |
| 6 | `whole` | `interpreter` vs `constant`, **DISAGREE subset only** |

Holm–Bonferroni over all six, α = 0.05. **Comparison 1 is the primary.**

## Gate

**CONFIRMED** — 1 rejects with a positive delta, and 2 rejects. The interpreter
states a whole causal claim about B that survives execution more often than one
assembled without reading B.

**PARTIAL** — 1 does not reject but 3, 4 or 5 does. Some clauses are read and the
conjunction is not; reported clause by clause with no headline.

**FAIL** — neither 1 nor any of 3–5 rejects.

**VOID** — 2 fails to reject while 1 rejects, meaning the design cannot separate
the interpreter from a coin.

## Not allowed after results

Adding arms, adding comparisons, promoting `analytic_executed` into the family,
scoring direction or magnitude against the true cause instead of the named one,
or reporting `whole` without the clause breakdown.

---

## Amendment 1 — the same design on a language target, declared before it runs

The design above is unchanged. It is applied a second time with the target model
class swapped, because "transfers between different target models and tasks" is
part of the claim this project is testing and every language result in the
repository was scored against the simulator's label.

| | |
|---|---|
| target | `entail/lm` — a decoder-only language model, 4 causes |
| why this domain | its carriers are disjoint, so pointer→name recovery has a ceiling of 1.0000; on the margin > 0.10 states, 34.2% of them are ones where B relies on something other than what drives the world, 39.8 per model |
| why not `agreement` | `SUBJECT_NUMBER` and `COLLECTIVE` share carrier position 2, so a pointer cannot separate them **by construction**; recovery ceiling is 0.9016 |
| why not `polarity` | only 2.8% disagreement, 3.3 states per model — too thin to pair over |

Family, gate and metrics are exactly as declared above. Results are reported for
the language target separately and are never pooled with the tabular one.

**A limitation of the method, recorded here because this run exposed it:** the
pointer identifies a cause only as precisely as the causes are separated in the
input. Where two causes share a carrier, they are indistinguishable to it, and
the ceiling is below 1 regardless of how well the model reads B. On tabular
domains the carriers happened to be disjoint, which is why this did not surface
before.
