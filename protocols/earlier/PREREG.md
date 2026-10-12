# Pre-registration — clean rerun of the localisation core

Frozen at commit `c306978` (the audited baseline) before the run named
`rerun_core`. Nothing below may change after results are seen. Anything that
does change afterwards is an exploratory run and must be labelled as one.

## Why this run exists

Four audits invalidated the transformer and language-model numbers of every
earlier run: interventions collapsed a unit to a scalar and broadcast it back,
so the sufficiency denominator was unreachable on any transformer and the
strongest non-learned control was arithmetically handicapped there. The
instrument is fixed. This run re-measures the localisation core with it, under a
declared protocol, and decides whether the learned interpreter earns its place.

## Condition

**Primary: `strict`.** The interpreter does not receive the candidate locus
profile or the retrieval posterior. Both are executed-intervention measurements
on the target model's own reference states; admitting them makes the setting
few-shot and transductive, and a trivial lookup over them beat every learned arm
in the audit. The `assisted` condition, in which every method including the
baselines receives both, is a separate declared run and is not reported here.

## Populations

| | |
|---|---|
| cells | `skirmish/mlp` (source), `skirmish/tab_transformer`, `clinic/mlp`, `agreement/lm` |
| target models per cell | 24 (8 seeds × 3 leak strengths) |
| fitted / held out | 16 / 8 |
| interpreter seeds | 0, 1, 2 |
| focal states per model | 160, of which 60 are reference and 100 are scored |
| probe states | 512 |

`seqworld` is excluded. Its executed counterfactual wrote a token absent from the
training distribution, so every measurement in that cell carried an
untrained-embedding shock. The domain is fixed but the old cell takes no part in
any new conclusion.

## Arms

| arm | what it is |
|---|---|
| `A_full` | the interpreter, fitted on the source cell |
| `A_output_only` | same capacity and training, all three internals channels closed (unit descriptors, locus, retrieval) |
| `A_ceiling_<cell>` | fitted in-cell; an upper reference, **not** a ceiling — a trivial method exceeded the old one |

## Baselines, declared in full

`first_order` (component-exact), `gradient`, `grad_x_act`, `activation`,
`correlation`, `weight_norm`, `random`, `oracle_effect`. Under `assisted` the
suite additionally contains `locus_retrieved` and `locus_oracle_name`; they are
not part of this run.

## Metrics

**Primary:** `executed.sufficiency@3` — the share of B's response to the executed
counterfactual reproduced by patching the three proposed units, now measured with
component-wise interventions so the denominator is attainable.

**Reported beside it, always, never selectively:** `executed.comprehensiveness@3`
(lower is better), `mediation.spearman`, `decoy_rejection`, and the diagnostic
`causal.spearman`.

Metric names: `sufficiency` and `comprehensiveness` here are *recovery fractions*
and carry the opposite sign convention to DeYoung et al. (2020), from whom the
names are borrowed. This is stated wherever they appear.

## Declared comparisons

Every one of these is evaluated on every cell and corrected as a single family
(Holm-Bonferroni, α = 0.05):

| # | metric | method | reference |
|---|---|---|---|
| 1 | `executed.sufficiency@3` | `A_full` | `first_order` |
| 2 | `executed.sufficiency@3` | `A_full` | `grad_x_act` |
| 3 | `executed.sufficiency@3` | `A_full` | `A_output_only` |
| 4 | `executed.sufficiency@3` | `A_full` | `random` |
| 5 | `decoy_rejection` | `A_full` | `first_order` |
| 6 | `decoy_rejection` | `A_full` | `correlation` |
| 7 | `mediation.spearman` | `A_full` | `first_order` |
| 8 | `causal.spearman` | `A_full` | `first_order` |

Tests are paired over held-out target models (each method scored on the same
models), two-sided exact sign-flip. With 8 held-out models the smallest
attainable p is 0.0078, so the design can reach significance; with the 4 models
of the earlier runs it could not, at any effect size.

Metrics are averaged per target model across the three interpreter seeds before
the test, so the bootstrap and the test unit stays the target model while seed
variance is folded into each model's estimate. Per-seed spread is recorded as
`seed_sd.<metric>`.

## Decision gate

Declared now, applied to the output without renegotiation.

**Pass** — the learned interpreter carries additional information: comparison 1
rejects (Holm-adjusted p ≤ 0.05) with a positive delta in at least one of the
three non-source cells, **and** `A_full` does not lose comparison 1 with an
adjusted p ≤ 0.05 in any cell. Then proceed to CS2 as the human-verifiable
semantic source.

**Fail** — `first_order` matches or beats `A_full` on the primary metric across
cells. Then the localisation layer is treated as solved analytically: the
project keeps `first_order` as its lower causal layer and the scientific question
moves up, to causal state → logical explanation, where the audit showed the
learned component is doing something a first-order term cannot express (naming a
cause, predicting the signed size of its effect). **A fail is not a reason to
discard the project**; it is a reason to stop claiming the lower layer.

**Ambiguous** — comparison 1 does not reject anywhere and `A_full` does not lose.
Then the honest report is that at this scale the two are indistinguishable, and
the next step is more target models, not a new metric.

## What a result must carry

Every result file records the commit hash, whether the tree was dirty, the config
path, the target seeds and leak strengths, the interpreter seeds, the cache
version and the torch version. A result without these is not citable.

---

# Amendment 1 — the first run's analysis plan was underpowered by construction

**Written after seeing `rerun_core`, before running `rerun_core2`. The
confirmatory analysis below therefore runs on fresh target populations
(`data_seed: 1`), not on the models already looked at.**

## The defect

The declared family was 8 comparisons × 4 cells = 32, with 8 held-out models per
cell. The smallest two-sided p a paired sign-flip test can return at 8 models is
`2 / 2^8 = 0.0078`; Holm-Bonferroni multiplies the smallest p by the family
size, so the smallest attainable adjusted p was `32 × 0.0078 = 0.25`. **No
comparison could reject at α = 0.05 at any effect size.** The gate could only
ever return AMBIGUOUS, and it did.

This is an error in the analysis plan, not a property of the data. `rerun_core`
is retained as an exploratory run; nothing from it is reported as confirmatory.

## The correction

Hold out **12 of 24** models per cell instead of 8, leaving 12 to fit. Then the
smallest attainable p is `2 / 2^12 = 0.00049` and the smallest attainable
adjusted p over the same 32-comparison family is `0.0156`, so α = 0.05 is
reachable. The family, the metrics, the arms, the baselines, the condition and
the decision gate are **unchanged** — only the split and the data seed move.

| | `rerun_core` (exploratory) | `rerun_core2` (confirmatory) |
|---|---|---|
| data seed | 0 | **1** — fresh target models |
| fitted / held out | 16 / 8 | **12 / 12** |
| smallest attainable adjusted p | 0.25 | **0.0156** |
| everything else | — | identical |

## Why fresh targets

The corrected family was chosen after seeing `rerun_core`'s numbers. Re-analysing
the same models under a family sized to let them reject would be selecting the
test after the result. New target populations under a new data seed make the
confirmatory claim a prediction rather than a re-description.

## What `rerun_core` is allowed to be used for

Descriptive statement of the pattern, labelled exploratory, with uncorrected p
values shown as such. It may motivate a hypothesis; it may not support a claim.
