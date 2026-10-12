# Pre-registration — semantic transfer, confirmatory

Frozen before `semantic_confirm` runs. Informed by the exploratory probe
(`results/semantic_explore.log`, and the corrected blind control in
`results/semantic_blind_fix.log`), which is what an exploratory run is for. Runs
on **fresh target populations** (`data_seed: 2`).

## Question

The localisation gate failed: with the first-order term computed correctly it
solves unit ranking on a language model (Spearman 0.995). Localisation has
therefore left the learned model. What remains is the step a first-order term
cannot take, because it is a property of a unit and not of a factor:

> Given the factual causal pattern over a model's units, and candidate causes
> described only from a disjoint set of reference states, can an interpreter name
> which cause is in play — and does that ability carry to a model trained on a
> different logical task?

## Design

| | |
|---|---|
| source | `agreement/lm` (in-cell naming ceiling 0.90) |
| transfer target | `polarity/lm` (in-cell ceiling 0.85–0.89, measured) |
| negative control cell | `skirmish/mlp` (in-cell ceiling 0.18 — nothing should work there) |
| condition | `strict`: no activation-patching result is ever an input |
| targets per cell | 24, **12 fitted / 12 held out** |
| interpreter seeds | 0, 1, 2 |
| smallest attainable paired p | 0.00049 |
| declared comparisons | 4 × 3 cells = 12 |
| smallest attainable adjusted p | 0.0059 — α = 0.05 is reachable |

Training budget is set to 20 epochs because the probe reached a training loss of
0.047 at 25, which is near-memorisation of 1200 problems from 12 models. This is
declared now, not tuned later.

## Arms and baselines

| | |
|---|---|
| `A_semantic` | causal pattern + candidate signatures + candidate locus |
| `A_no_locus` | the same without the locus profile |
| `A_blind` | architecture shape and B's own output only — **both** the per-unit first-order features and `log_fo_mass` masked. The probe's blind arm leaked through the second channel and its numbers are void |
| `prior`, `largest_effect`, `random` | strict-condition baselines |
| `retrieval` | reference-state vote in causal-pattern space. **Assisted condition**: it uses ground-truth annotations of the target model's reference states. Reported always, never as the headline comparator |
| in-cell ceiling | an interpreter fitted on the transfer cell itself |

## Metric

**Primary: `naming.balanced_accuracy`** — mean per-class recall. Plain accuracy
is reported beside it always. Sufficiency is not a semantic score and is not
reported here.

## Declared comparisons

`A_semantic` against `A_blind`, `prior`, `random` and `A_no_locus`, on balanced
accuracy, on every cell, paired over held-out target models, two-sided exact
sign-flip, corrected as one family of 12 (Holm-Bonferroni, α = 0.05). Metrics are
averaged per target model across the three interpreter seeds before testing.

## Gate

**Pass** — `A_semantic` beats `A_blind` on `polarity/lm` with adjusted p ≤ 0.05
and a positive delta. The mapping from causal pattern to named cause transfers to
a model trained on a different logical task, and the next step is natural text.

**Source-only** — it beats `A_blind` on `agreement/lm` but not on `polarity/lm`.
The mapping generalises across independently trained models of the same task and
not across tasks; the claim is scoped to that and the next question is what the
task boundary is made of.

**Fail** — it beats `A_blind` nowhere. Naming does not transfer under the strict
condition, and the honest report is that only the assisted condition works, where
a parameter-free reference lookup already reaches 0.96.

**In every branch**, if `A_semantic` beats `A_blind` on `skirmish/mlp` — the cell
whose in-cell ceiling is 0.18 — that is evidence of a leak, not of transfer, and
the run is void pending investigation.

## Not allowed after results

Changing the metric, the family, the arms, the gate or the condition. Any such
change makes the run exploratory and it must be labelled so.
