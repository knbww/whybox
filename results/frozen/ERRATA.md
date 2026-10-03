# Errata

Four independent audits of this repository — of the numbers, the measurement
code, the citations, and the experimental design — were run on 2026-08-21.
The frozen results in this directory are kept as a historical record. This file
lists what in them is wrong, what is unaffected, and what must not be cited.

Nothing here is a matter of interpretation: every item was reproduced.

## 1. INVALIDATING — the intervention instrument is degenerate on transformers

A "unit" is summarised as a **scalar** (mean over an attention head's positions
and head dimensions, or over an FFN neuron's positions), and an intervention
writes that scalar back to every component. On an MLP this is exact — there is
one component. On any transformer it discards the unit's internal structure.

Reproduced on `agreement/lm`, patching **every** unit to its counterfactual
value and reading the fraction of B's response recovered:

| decisive factor | scalar patch (as published) | structure-preserving patch |
|---|---|---|
| SUBJECT_NUMBER | 0.920 | 1.000 |
| COORDINATION | 1.101 | 1.000 |
| QUANTIFIER | **0.210** | 1.000 |
| COLLECTIVE | **0.453** | 1.000 |

Consequences:

- **`sufficiency@k` denominators are unreachable on transformer and LM cells.**
  The metric normalises against B's full response, which no unit patch could
  attain: mean 0.72 on `agreement/lm`, 0.37 on the worst held-out model. Every
  cross-cell comparison in `docs/RESULTS.md` therefore compares MLP cells on a
  0–1 scale against LM cells on a 0–0.72 scale. **Do not cite the cross-cell
  sufficiency table.**
- **"Patch-based validation is blind on language models" is wrong.** It was
  attributed to the target models carrying factors through a shared circuit. It
  is an artifact of the instrument: under a structure-preserving patch,
  carrier selectivity on the transformer cells rises roughly thirtyfold and
  exceeds the MLP cells. **Retract that section and the `carrier_selectivity`
  table.**
- **The LM circuit result was understated.** The published trace reports three
  attention heads reproducing 47% of the grammatical response; that is a
  measurement floor, not the model's behaviour.
- `first_order` — the strongest non-learned control — is arithmetically wrong
  for multi-component units (it computes `(ā − c)·Σg` instead of `Σ g_j(c − a_j)`),
  which handicaps it on exactly the cells the transfer claim rests on.

Unaffected: everything measured on `skirmish/mlp` and `clinic/mlp`, and every
naming/balanced-accuracy result, which does not route through a patch.

## 2. INVALIDATING — a decisive baseline was missing

Ranking units by the mean measured indirect effect of the retrieved factor over
reference states — no learned parameters — beats the interpreter, its in-cell
"ceiling", and the oracle on every transformer and language cell:

| cell | locus baseline | interpreter | "ceiling" |
|---|---|---|---|
| skirmish/mlp | 0.251 | 0.392 | — |
| clinic/mlp | 0.334 | 0.405 | 0.478 |
| skirmish/tab_transformer | **0.448** | 0.343 | 0.377 |
| clinic/tab_transformer | **0.342** | 0.279 | 0.376 |
| seqworld/transformer | **0.601** | 0.513 | 0.525 |
| agreement/lm | **0.603** | 0.433 | 0.508 |

A quantity a trivial method exceeds is not a ceiling. The direction of the
published finding inverts: the learned interpreter adds value on the small
tabular networks and is dominated by a lookup elsewhere.

## 3. INVALIDATING — a table with no source

The section "Retrieval plus a learned reranker, measured in-cell" in
`docs/RESULTS.md` reports twelve numbers, eleven of which match no row of any
results file. They came from an ad-hoc script that was never persisted, and the
stated setup ("16 models to fit, 8 held out") is wrong — it was 8 fit and 4
held out. **The table has been removed.** The claim it supported (that a learned
reranker beats retrieval in-cell) must be re-run and frozen before it is used.

## 4. The information budget contradicted itself

`docs/PROTOCOL.md` stated that intervention results are targets and never
inputs. The interpreter was nonetheless given the candidate locus profile — an
averaged indirect-effect measurement — and the retrieval posterior, built from
ground-truth annotations on 60 reference states of each held-out model. No
baseline received either.

Consequently **"zero-shot" was the wrong word** for the naming results: they are
few-shot and transductive. Later work declares two conditions explicitly — a
strict condition with neither channel, and an assisted condition where every
method including the baselines receives both.

## 5. Statistics

- With 4 held-out models the smallest attainable two-sided paired p is 0.125.
  No result in `pilot.json`, `llm_transfer.json` or `explanation.json` can reach
  nominal significance by any paired test, before correction.
- Roughly 7,000 method-by-cell-by-metric comparisons were made across the runs
  with no multiplicity correction anywhere.
- The percentile bootstrap in `_bootstrap_ci` has ~80% coverage at n=4, so the
  published intervals are roughly a quarter too narrow.
- Every interpreter was trained with a single seed, so arm differences of
  0.01–0.07 are not separable from initialisation noise.
- The held-out model indices are identical across cells, so "in all six cells"
  is six correlated observations, not six independent confirmations.

## 6. Selective reporting

- `comprehensiveness@k` was defined as half of the headline metric and never
  reported. On `agreement/lm` the interpreter is 2.5× worse than the in-cell
  arm on it.
- `A_output_only` beats `A_full` on three of four transfer cells in the semantic
  run's naming table and was omitted from that table.
- `explanation.score@3` for the interpreter arms was not reported on the two
  cells where the metric was declared valid; on `clinic/mlp` it is below `prior`.
- `effect.sign_accuracy` on `entail/lm` is 0.405 — below a coin flip — and was
  not reported alongside the favourable `effect.rho`.

## 7. Specific numerical corrections

| claim | published | correct |
|---|---|---|
| `A_output_only` at or below random | four of six cells | **three of six** |
| gradient-family ablation cost on LM | 0.230 against 0.417 | **0.230 against 0.353** (0.417 is a different arm) |
| clinic ceiling fraction, attributed to the pilot | 84% | **82.5%** for the pilot; 84% is `A_multiarch` in a later run |
| `first_order` rank correlation on MLPs | ρ ≈ 0.99 | **0.980–0.985** |
| MAN_ADVANTAGE share of decisive states | 58% | **59.1%** |
| test count in README | 35 tests, ~5 s | **64 tests, ~8 s** |
| unit range in the published brief | 48–300 | **44–300** overall; 48–150 for the CS2 models shown |

## 8. Attribution

The repository had no bibliography while using established terminology. Two
metric names, **sufficiency** and **comprehensiveness**, are taken from the
rationale-faithfulness literature (DeYoung et al., ACL 2020) and were used with
the **opposite sign convention**, uncited. Causal mediation and `do(·)` notation
(Pearl 2001; Vig et al. 2020), activation patching (Meng et al. 2022), causal
abstraction (Geiger et al. 2021), the logit-difference readout and IOI (Wang et
al. 2023), agreement attraction as an LM syntax probe (Bock & Miller 1991;
Linzen et al. 2016), Set Transformer (Lee et al. 2019) and weight-space
equivariant networks (Navon et al. 2023; Zhou et al. 2023), and the automated
interpretability line this work sits in (Bills et al. 2023; Schwettmann et al.
2023; Shaham et al. 2024) all require citation before any of this text is
published.

## 8b. The semantic confirmatory PASS is missing its strongest baseline

`semantic_confirm.json` and the tag `semantic-confirm-pass` record a
pre-registered PASS: the interpreter names the decisive cause in models trained
on a different logical task at 0.527 balanced accuracy against 0.348 for the
internals-blind control (+0.179, adjusted p 0.006). That comparison was declared
in advance and it holds.

**It is not the comparison that matters.** A one-line analytic baseline —
the gradient of the output with respect to the input embedding, dotted with
(neutral value − current value), summed per position — was never in the suite.
Paired on the same twelve held-out models and the same states:

| cell | input first-order | A_semantic | A_blind | delta | p |
|---|---|---|---|---|---|
| `agreement/lm` *(source)* | 0.836 | **0.926** | 0.725 | +0.090 | 0.0039 |
| `polarity/lm` *(transfer)* | **0.894** | 0.527 | 0.348 | **−0.367** | 0.0005 |

The learned interpreter beats the formula where it was fitted and loses to it by
0.367 on the cell that carried the transfer claim. The formula is *better* on the
transfer cell than on the source, which is what a method with no learning should
look like: it has no source/target asymmetry at all.

**The claim "the mapping from causal pattern to named cause transfers" is
therefore not supported in the sense that matters.** What transfers is beaten by
calculus. This is the same failure the localisation gate exposed, one level up,
and it was found by asking the question the audit taught rather than by a new
audit.

The baseline is now in the suite (`mint.semantic.run.input_first_order`). The tag
is left in place because history should not be rewritten; this entry is what
qualifies it.

**Correction (2026-09-06 audit).** The sentence above was false for as long as it
stood. `input_first_order` was defined and never called: its only two references
in the repository were its own `def` and this paragraph, so it was absent from
`baselines()` and from `scripts/semantic_confirm.py`, and re-running the
confirmatory script reproduced the *unqualified* PASS. It is wired in as of this
correction. Note also that `results/input_fo_vs_semantic.log` — the sole artefact
behind the table above — has no committed generator, which is the same defect as
item 3 of this file. The table must be regenerated and frozen before it is cited,
and no test against this baseline may enter the family without a new
pre-registration: it was not in the declared family of `semantic_confirm`.

## 8c. Whole-statement composition numbers are invalid for inference

`results/frozen/structure_confirm.json` is the confirmatory record and stands as
written; it is **not** recomputed. Its conclusion -- that the internals stream
adds nothing to transfer while a learned classification of the first-order
pattern does -- concerns balanced accuracy over structure types and is unaffected
by what follows.

The **composition** numbers reported for the generated-claim experiments
(`docs/RESULTS.md`, "Generating the whole claim") carried two defects and must
not be used for any final inference:

1. **Mixed regimes.** The instrument read sign and strength by *executing* its
   own proposed support, scoring 1.000 on both by construction, while the learned
   generator predicted them without a probe. A composition score spanning the two
   regimes compares an executed measurement against a prediction.
2. **The strength clause referred to a quantity the claim never states.** Truth
   was bucketed from the *full* response while verification patched the
   *proposed support*, which recovers 90% of it by construction. A bucket
   boundary falling between them cost even the oracle its own clause (0.62).

Both are fixed in `mint.generative.composite`: sign and strength are now defined
by the effect of the support a claim proposes, and `formula_template` takes an
explicit regime flag so both sides can be scored on matched terms. The clause
accuracies (type, support, sign, strength taken separately) are unaffected by
defect 1 and by defect 2 only in the strength column.

## 9. What survives the audit

- The intervention *mechanics*: edits apply to the right tensors on the right
  axis in every model class, no leakage across batch rows, gradients attached to
  the tensors the edits touch, the LM's causal mask and token contrast correct.
- Train/evaluate separation at the level of target models, verified for all runs.
- Reference states used to build candidate descriptions are disjoint from the
  states scored, verified by scrambling.
- Permutation equivariance over units and over candidates.
- Every table cell in `docs/RESULTS.md` matches the frozen JSON to the printed
  precision, except the unsourced table in item 3.
- The negative results — that naming did not transfer from tabular sources, and
  that a learned mapping alone underperforms retrieval — are unaffected by items
  1 and 2, and item 2 strengthens them.

## 10. The published brief

`docs/brief/index.html` was published as a shareable page. Its headline claims
came from the runs items 1 and 2 retract; it additionally attributed its numbers
to the wrong config, drew two wiring arrows unsupported by measurement, and
mislabelled the correlation bar's sample size.

**The brief has been deleted (2026-09-06), not merely annotated.** An earlier
version of this entry said it had been "withdrawn and replaced with a notice";
that was inaccurate — `index.html` and `notice.html` were added in the same
commit, so nothing was ever replaced, and because `index.html` is the
conventional directory index any static host served the retracted page and never
the notice. Both files are now removed. Nothing in the repository presents those
numbers as current.

## 11. The twelve held-out targets are four independent seeds, not twelve (2026-09-14)

Applies to `follows_b.json`, `transfer_b.json`, `statement.json` and
`statement_lm.json`. Found by a full-system audit and reproduced independently
from the committed files.

Each target population is an 8 × 3 factorial: eight training seeds, each trained
at three decoy strengths (0.0, 1.0, 2.0). Fitted models are seeds 0–3; held-out
models are seeds 4–7, in all three populations used (`skirmish`, `clinic`,
`entail`). Train/test separation is therefore clean at the seed level as well as
the model level. But the three models sharing a seed share an initialisation and
a data stream, so they are not twelve exchangeable units.

Every protocol in this line states "smallest attainable paired p 0.000488",
which is 2/2¹² and assumes twelve. Treating the seed as the unit:

| primary comparison | Δ | per-model Δ > 0 | p (n = 12) | seed groups with Δ > 0 | p (n = 4) |
|---|---|---|---|---|---|
| `follows_b`, vs constant, DISAGREE | +0.2672 | 10 / 12 | 0.00293 | **4 / 4** | 0.125 |
| `transfer_b`, skirmish→clinic, DISAGREE | +0.1393 | 9 / 12 | 0.00586 | **4 / 4** | 0.125 |
| `transfer_b`, clinic→skirmish, DISAGREE | +0.4271 | 11 / 12 | 0.00098 | **4 / 4** | 0.125 |
| `statement`, whole vs constant | +0.2499 | 12 / 12 | 0.00049 | **4 / 4** | 0.125 |
| `statement_lm`, whole vs constant | +0.2899 | 12 / 12 | 0.00049 | **4 / 4** | 0.125 |

**How to read it.** 0.125 is the floor at n = 4, and every primary reaches it:
the effect is positive in every independent seed group, which is the strongest
evidence a four-seed design can express. It is *not* absence of an effect. But
no comparison would survive Holm over its family of six at the seed-clustered
level (0.125 × 6 = 0.75), so **"adjusted p 0.0029" overstates the resolution of
the design and must not be quoted without this qualification.** The audit
measured the intraclass correlation of the paired differences by seed at −0.41 to
+0.59 and negative in about half the comparisons, so the three models per seed
behave more like distinct targets than replicates, and the true effective n lies
between 4 and 12.

**The defensible sentence:** *the effect is positive in all four independent seed
groups and in 9 to 12 of the 12 held-out models, for every primary comparison.*
A future run that wants the resolution the protocols claimed needs twelve
independent seeds rather than four seeds at three decoy strengths.

## 12. Other corrections from the same audit

- The `agreement` recovery ceiling quoted as **0.9016** was computed without the
  protocol's own margin filter; under margin > 0.10 it is **0.8708**. The adjacent
  `entail` figures in the same table were filtered. The conclusion (exclude
  `agreement`) is unaffected.
- `recover_name` averages a pointer's scores over a cause's carriers. That choice
  was never justified and it handicaps the parameter-free analytic pointer on
  two-carrier causes; summing instead lifts that pointer from 0.6264 to 0.7371 on
  `skirmish` ALL, above the interpreter's 0.7268. On the DISAGREE subset averaging
  is the better rule for the analytic arm and the interpreter's lead survives. No
  gate is affected — the analytic pointer is outside every family — but the
  statement that the interpreter out-reads the derivative on pooled accuracy
  depends on this choice.
- "The interpreter does not see the candidate mask" is not literally true: `fo` is
  exactly zero at every non-candidate position and `sign(fo)` is an input feature,
  so the mask is recoverable. Its measured worth is about the constant floor
  (a mask-only pointer: 0.4999 on `skirmish` ALL, against the constant's 0.6099).
- The magnitude-clause confound recorded for `statement_lm` applies to `statement`
  as well: each arm is scored against the intervention *it* named, so an arm that
  names one fixed cause faces an easier sub-problem. It produced `statement`'s
  favourable +0.1337 as surely as `statement_lm`'s unfavourable −0.2401. Conditional
  on the cause being right — where every arm faces the same intervention — the
  interpreter scores 0.627 on `skirmish` (analytic 0.785) and 0.580 on `entail`
  (analytic 0.345).
- `follows_b.json` records commit `31556cd` with `dirty: true`; the code that
  produced it, with the corrected six-comparison family, was committed in
  `d002ead`. A re-run at that commit reproduces the file byte-identically.
