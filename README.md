# whybox

**Can one neural network read another network's internal signals and name, in human
terms, the cause of its decision?**

A student research project on causal interpretability. A trained interpreter **A**
receives only internal machine-learning signals of a target model **B** (gradient-based
estimates of how each input and each internal unit moves B's answer). From them it names
the cause of B's decision in the task's own terms: *price*, *comfort* or *safety* for a car;
*the number of the subject* or *the number of a distracting noun* for a sentence. Every
answer is checked by an intervention on B: the named cause is suppressed and B's answer
is measured again.

```
target model B ──> B's internal signals ──> interpreter A ──> "the cause is SAFETY"
                                                                     │
                                       checked on B: do(SAFETY := neutral) ──> B's answer moves most
```

The correct answer is **the cause B itself relies on**, measured by intervening on B,
not the cause that drives the outcome in the world. The two differ whenever B has learned
something other than the true rule, and those states are reported separately: a predictor
of the world is wrong on all of them by construction.

**Try it on Colab:**
[`colab/sentences_tricks_colab.ipynb`](https://colab.research.google.com/github/knbww/whybox/blob/main/colab/sentences_tricks_colab.ipynb)
runs the saved interpreter on GPT-2 XL, Qwen3-8B, Qwen3-14B and Qwen2.5-7B (the last three in
4 bits), the two 64-segment diagnostics and the integrated-gradient control described below.
Each run writes a new `results/my_*.json`; choose a GPU runtime. The notebook's notes are in
Russian. It was re-targeted from Pythia, SmolLM2, OLMo-2 and Phi-2 to these models before any
of those families was run, so no results exist for them.

**Weights on Hugging Face:** [la-Ilyaso/whybox](https://huggingface.co/la-Ilyaso/whybox).

## Why this matters

A model can be accurate for the wrong reason. In a study of pneumonia detection on chest
X-rays from several hospitals, a separate network identified the hospital a scan came from
almost perfectly, and pneumonia models trained on hospitals with different disease
prevalence did worse on scans from other hospitals (Zech et al., 2018). An image classifier
recognised horses by a copyright tag (Lapuschkin et al., 2019). Accuracy alone does not
reveal this: one has to know what a decision rests on. Attribution methods answer that per
model and per decision, as an importance map. This project asks whether one interpreter,
trained once, can name the cause of a decision for models it has never seen, with every
answer checked by an intervention on the model, and whether the skill transfers across
models and tasks.

## The hypothesis

> If one AI model is trained to understand the internal causal state of another model
> from universal ML signals, it can give a sufficiently accurate, human-understandable
> causal interpretation that transfers between different target models and tasks.

Its parts, each judged on its own: (1) one trained interpreter; (2) it reads universal
signals of B; (3) it names the cause B itself relies on, not the world's; (4) accurately
enough: above chance, a constant answer and a predictor of the world; (5) in human terms;
(6) across target models and tasks it was not trained on. Each result below says which
parts it supports, leaves untested or contradicts.

"Universal" is meant narrowly: the signals are computed the same way for any differentiable
model whose activations and gradients are accessible, and after normalisation their format
does not depend on the model's architecture or size. That the interpreter transfers does not
follow from this; it is tested.

## Results

### Simulated worlds, twelve independent initialisations (36 held-out models)

A tactical game world with seven causes. Each initialisation contributes three held-out
models (three strengths of a decoy feature), averaged. Protocols committed before the runs:
[`scripts/seeds12_all.py`](scripts/seeds12_all.py),
[`scripts/typical_control.py`](scripts/typical_control.py),
[`scripts/pointer_control.py`](scripts/pointer_control.py).

| method | all states | B departs from the world (12% of states) | B departs from what 36 other models rely on (12%) |
|---|---|---|---|
| interpreter A [95% CI] | **73.8%** [69.3, 78.1] | **59.3%** [53.9, 65.1] | **54.9%** [50.9, 59.2] |
| constant | 63.2% | 28.1% | 28.0% |
| perfect predictor of the world | 87.7% | **0%** | n/a |
| predictor of typical models | 88.4% | 23.5% | **0%** |
| inputs-only classifier | 86.7% | 25.0% | 20.6% |
| first-order pointer (control, see below) | 87.8% | 65.8% | 61.9% |
| chance | 14.3% | 14.3% | 14.3% |

This supports parts 1 to 4: one trained model reads B's signals and names B's own cause
where B departs from the world or from the typical model, above chance, a constant and the
predictors that never read B. All six comparisons of the main protocol reject the null
after Holm correction: adjusted p = 0.003 for five of them, and 0.012 for the comparison
with the constant where B agrees with the world. The three comparisons of the strict check
(last column) are at adjusted p = 0.003. The smallest unadjusted p attainable with twelve
units is 0.0005.
**Caveat:** on all states, predictors that never read B are more accurate (typical models
88.4%, inputs only 86.7%; the latter significantly, adjusted p = 0.003), because models
mostly rely on the same cause. The pooled accuracy alone does not show that B is read; the
cells where B departs do.

The **first-order pointer** is a parameter-free control. It sums the same per-input
estimates over the inputs of each cause, so it needs the grouping of inputs into causes,
which the interpreter does not receive. Where the pointer is wrong (542 of 4450 states), the
interpreter is right in 20.8% of them, and in 29.9% where B departs from the world.

**Which signals matter.** A diagnostic without a frozen protocol removed the stream of
internal-unit signals, or the global signals, in three worlds: nothing changed detectably
(adjusted p ≥ 0.41). The per-input estimates carry the accuracy.

**Dynamic signals** ([`scripts/dynamics_deep.py`](scripts/dynamics_deep.py)). Adding all
dynamic signals, computed along the path to the neutral state, raised accuracy on all states
to 78.5% (+4.7 points, adjusted p = 0.005). Integrated gradients alone give 78.3% (+4.5 points,
adjusted p = 0.005). Where B departs from the world or from typical models there was no
detectable change (+1.1 and +1.2 points, adjusted p = 1.0). On these small models the number
of path segments (4 to 64) makes no detectable difference.

### A real dataset: UCI Car Evaluation

1728 cars, six attributes, the published four-level acceptability, used as published.
The causes are PRICE (buying and maintenance price) and COMFORT (doors, persons, luggage
boot), two concepts of the published decision hierarchy, and the attribute *safety*, which
together with COMFORT forms the hierarchy's TECH concept (chance 33.3%). Twelve independent
held-out initialisations, 14 comparisons under one Holm correction, protocol committed
before the run: [`scripts/car_study.py`](scripts/car_study.py), result
[`results/car_study.json`](results/car_study.json).

| question | interpreter | references | read-out |
|---|---|---|---|
| names the cause B relies on | **68.5%** | constant 38.7%; inputs-only classifier 93.0%, typical models 93.5%; pointer 66.1% | supported |
| where B departs from the published rule (0.85% of states) | **47.9%** | constant 17.0%, world 0%; pointer 49.0% | supported |
| where B departs from what other models rely on (6.8%) | 39.0% | constant **48.9%** (significantly higher, adjusted p = 0.009); inputs only 20.2%; pointer 40.3% | **not supported** |
| trained only on the simulation, reads Car models | 60.3% (all states) | constant 38.7% | not supported: 37.3% vs 48.9% in the cell above |
| trained only on Car, reads simulation models | 61.4% (all states) | constant 60.1% | not supported on all states; where B departs, 53.5% and 51.5% vs about 25% |
| dynamic signals | **85.6%** vs 68.5% | | better |

On real data, predictors that never read B reach 93% on all states (the inputs-only
classifier significantly above the interpreter, adjusted p = 0.007). So the evidence of
reading B rests on the second row, 389 states. A post-hoc analysis (in
[`scripts/pointer_control.py`](scripts/pointer_control.py)) found that 313 of those 389
states come from models of one architecture. There, a predictor of typical models of the
same architecture reaches 55.3%, against 53.0% for the interpreter. The strict check is
significantly *below* a constant, which contradicts part 3 ("this model, not models in
general") on real data, and neither transfer supports part 6.

Post hoc, the shortfall sits in states where the published rule names no single cause:
2741 of the 3130 strict-check states, 38.2% vs 46.0% for the constant. On the remaining 389
states the interpreter scores 50.2% against 11.8%. The gap between B's top two causes on
these states is small (median 0.20 output standard deviations vs 1.30 elsewhere). It is just
as small on tactical models (0.19), where that cell is read, so near ties do not explain
the failure.

### Pretrained language models, full sentences with a trick

The interpreter was trained only on twelve toy language models of 19 to 59 thousand
parameters and is applied without further training. B chooses between *is* and *are* after
sentences with a relative clause, where a second noun pulls the verb's number the wrong way:
"The quiet nurse that the critics like ...". Causes: the number of the subject, the number
of the distracting noun, the adjective; a long template has seven causes. 160 sentences are
built per model and template; the 119 to 142 that pass the margin rule are scored. The floor
is a random choice among the causes whose suppression changes B at all. Protocol:
[`scripts/sentences_tricks.py`](scripts/sentences_tricks.py). Here the interpreter reads
only the per-position integrated gradients along a 16-segment path; the unit and global
streams are zeroed.

| model | precision | object clause | subject clause | long template |
|---|---|---|---|---|
| GPT-2 (124M) | fp32 | **1.000** / 0.563 | **0.928** / 0.548 | **0.886** / 0.301 |
| GPT-2 large (774M) | fp32 | **0.961** / 0.550 | **0.866** / 0.552 | **0.652** / 0.309 |
| GPT-2 XL (1.5B) | fp32 | **0.969** / 0.534 | **0.906** / 0.551 | **0.736** / 0.307 |
| Qwen2.5-0.5B-Instruct | fp32 | **0.977** / 0.562 | **0.919** / 0.579 | **0.792** / 0.308 |
| Qwen2.5-1.5B-Instruct | fp32 | **0.962** / 0.582 | **0.883** / 0.567 | **0.790** / 0.313 |
| Qwen2.5-3B-Instruct | fp16 | **0.954** / 0.571 | **0.915** / 0.581 | **0.818** / 0.308 |
| Qwen2.5-7B-Instruct | 4-bit | **0.944** / 0.578 | **0.882** / 0.568 | **0.808** / 0.326 |
| Qwen3-8B | 4-bit | **0.905** / 0.603 | **0.816** / 0.608 | **0.735** / 0.346 |
| Qwen3-14B | 4-bit | **0.867** / 0.563 | **0.704** / 0.600 | **0.726** / 0.316 |

Interpreter / floor; every cell is above the floor after Holm correction. This supports
parts 1, 2, 5 and 6 of the hypothesis on one task, subject-verb agreement, in models up to
14 billion parameters. It supports part 4 in the sense of "above chance among the causes
that act": this protocol has no constant or world-predictor arm.

Part 3 is shown only descriptively. Models depart from grammar in 81 sentences over the
nine models: 70 in the subject clause, 7 in the object clause, 4 in the long template. On
these the interpreter names the model's own cause in 74.1% (56 of 70, 4 of 7 and 0 of 4),
against 39.1% for the floor and 0% for a predictor of grammar. This pooled figure is not a
pre-registered test.

The integrated-gradient, 16-segment recipe was chosen in earlier unregistered diagnostics
on GPT-2 and Qwen2.5-0.5B with other sentences, so those two rows are not untouched targets.

An integrated-gradient pointer, the same control as above, was computed on six of the nine
models. It did not fit in the free Colab GPU's memory for the three 4-bit models. It is
right in 91.3% of sentences vs 88.3% for the interpreter. In the 208 sentences where it is
wrong, the interpreter is right in 10.7%. On the 52 departures from grammar among those six
models, it is right in 57.7% vs 75.0% for the interpreter.

Accuracy is lower on the two Qwen3 models at 16 path segments. A post-hoc diagnostic, outside
the protocol's read-out, ran only these two models with 64 segments. The mean completeness
error fell 4 to 8 times, and accuracy rose to 0.957 / 0.917 / 0.827 (Qwen3-8B) and
0.948 / 0.917 / 0.835 (Qwen3-14B). The completeness error does not track accuracy across
families: GPT-2 large has an error of 0.55 and reads at 0.961. The other models were not
run with 64 segments.

### Earlier results, with a qualification

Before the twelve-initialisation design, held-out models came from four independent seeds
at three decoy strengths, so p-values that assume twelve units overstate the resolution;
every effect below is positive in all four seed groups.

- **Toy logic world.** A transformer language model: 71.4% (chance 25%, constant 31.6%).
- **Transfer between unrelated simulated worlds.** Trained only on medical models, the
  interpreter reads tactical ones at 70.5%; trained on the tactical models themselves it
  reaches 72.7%. The difference is not established (p = 0.075, outside the protocol). In the
  other direction it reaches 54.1% (chance 20%, the constant carried from the source world
  14.9%).
- **First-order pointer on the same targets.** 70.5% on the toy logic model, 88.7% on the
  tactical models and 74.2% on the medical ones.

## Limitations

- The causes and the inputs through which they act are listed in advance; the interpreter
  points at inputs and the name comes from that list. Two causes acting through the same
  input are indistinguishable, and a cause outside the list cannot be named.
- One cause per decision. States without a clear main cause (11% to 26% of states,
  depending on the task) are excluded.
- The interpreter reads gradient-based estimates of influence, not raw activations or
  weights, and the per-input estimates carry its accuracy. Direction and size of the
  effect are measured on B, not predicted by A.
- The correct answer depends on the neutral values chosen for each cause, and the input
  estimates are computed relative to the same values; the choice was not varied.
- "This model, not models in general" holds on simulated worlds and is significantly
  reversed on Car Evaluation; transfer between simulation and real data was not shown.
- Pretrained language models were tested on one task in three templates, and the departures
  from grammar are too few per model for a separate test.
- Path signals needed more segments for the Qwen3 models; a future protocol should fix the
  rule for choosing the number of segments in advance.

## How the work is checked

From the twelve-initialisation run (`seeds12_all`) on, every confirmatory run has its
protocol in the docstring of its script, committed before the run. The protocol states the
claim, the compared arms, the metrics, the family of comparisons and the read-out rule. The
script refuses to start if it differs from the committed version or if the tree is modified,
and refuses to overwrite its result.

The commit hashes recorded in result files refer to the author's private development
repository; this repository is an export without that history. What can be checked here is
the sha256 of each script, recorded in its result, and the times below. Every hash equals the
script in this repository, except for the first language-model run (GPT-2 and Qwen2.5-0.5B).
That run used the version before Amendment 1, which only added the `--load-in-4bit` and
`--steps` options. The Colab runs record commits of this public repository.

| run | protocol committed (UTC) | run started (UTC) |
|---|---|---|
| `seeds12_all` | 2026-09-22 13:37:17 | 13:37:25 |
| `typical_control` | 2026-10-03 06:14:08 | 06:14:09 |
| `dynamics_deep` | 2026-10-03 06:28:09 | 06:28:10 |
| `sentences_tricks` (GPT-2, Qwen2.5-0.5B) | 2026-10-03 07:31:54 | 07:31:54 |
| `car_study` | 2026-10-03 12:20:06 | 12:20:14 |
| `pointer_sentences` (GPT-2, Qwen2.5-0.5B) | 2026-10-04 11:24:49 | 11:25:01 |
| `pointer_control` | 2026-10-04 11:24:49 | 11:32:20 |

The earlier runs (`follows_b`, `transfer_b`, `statement`) predate this discipline. Their
protocols were separate documents that are not in this repository, and their scripts
overwrite their output. `follows_b.json` was produced from a modified working tree; a
byte-identical rerun is documented in [`results/frozen/ERRATA.md`](results/frozen/ERRATA.md)
§12, and §11 explains the four-seed qualification.

Statistics on simulated and Car models use exact sign-flip tests paired over independent
initialisations (all 4096 sign patterns for twelve units), with Holm correction over the
declared family and 95% percentile bootstrap intervals over initialisations. On pretrained
language models the unit is the sentence: a Monte Carlo sign-flip test (20,000 flips) over
the scored sentences of one model, with Holm correction over the (model, template) pairs of
each run.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" transformers
.venv/bin/python -m pytest -q
```

- **Pretrained language models:** the Colab notebook above, or locally
  `PYTHONPATH=.:src:scripts python scripts/sentences_tricks.py --models gpt2 --out results/my_gpt2.json`.
  This reproduces the GPT-2 row in about two minutes on a CPU; the other rows need
  `--device cuda` and, as in the table, `--dtype float16` or `--load-in-4bit`. The
  interpreter weights are in [`results/sentences_tricks_interpreter.pt`](results/sentences_tricks_interpreter.pt).
- **Simulated worlds and Car Evaluation:** these read cached target populations
  (`results/cache/`, about 5 GB, not in the repository). Builder scripts in `scripts/`
  recreate them (for example `fresh_build.py`, `volume_build.py`, `car_build.py`), and every
  consumer checks the sha256 recorded in the manifests under `results/`. A rebuild with other
  library versions may differ bit for bit, and then the check stops the run.
  A finished run refuses to overwrite its result; move the result file away to run it again.

## Data and models

- `data/uci/`: Car Evaluation (Bohanec & Rajkovič, 1988), Balance Scale (Siegler, 1976) and
  Tic-Tac-Toe Endgame (Aha, 1991) from the UCI Machine Learning Repository, CC BY 4.0, with
  their original description files.
- `data/wikidata/`: country facts from Wikidata (CC0), with the query and the fetch date, used
  by an earlier experiment (`src/mint/domains/wikifacts.py`).
- Pretrained models (GPT-2, Qwen2.5, Qwen3) are downloaded from Hugging Face under their own
  licenses and are not redistributed here.

## Related work

Probing classifiers show that a property is represented in a model's activations, not that
the model uses it (Alain & Bengio, 2016; Belinkov, 2022). Costarelli, Allen & Field (2024)
trained meta-models that read a language model's activations and answer questions about its
behaviour in natural language. In LatentQA (Pan et al., 2026) and Activation Oracles
(Karvonen et al., 2026), a copy of a language model is fine-tuned to answer open questions
about the activations of that model or its fine-tuned variants. Their correct answers come
from the source data: a system prompt or persona, a text label, or facts inserted by
fine-tuning. MAIA (Shaham et al., 2024) gives a vision-language model tools for experiments
on other models' components, to describe the features neurons respond to and to find failure
modes.

What is specific here:
- one interpreter is applied to models of different architectures and sizes;
- it names the cause of a particular decision;
- the correct answer is the cause the model itself relies on, defined by an intervention on
  that model, and every answer is checked by the same intervention;
- states where the model departs from the world or from typical models are scored
  separately.

## License

Code and the interpreter weights: [MIT](LICENSE). Text, figures and result files:
[CC BY 4.0](LICENSE-CC-BY-4.0.txt). Data keep their own licenses (UCI: CC BY 4.0;
Wikidata: CC0).

## Use of AI tools

Most of the code and the experiment scripts, and the drafts of this README and of the model
card, were written with an AI coding assistant (Claude, Anthropic) under the author's
direction. The research question, the hypothesis and the design decisions are the author's.

## References

- Alain G., Bengio Y. Understanding intermediate layers using linear classifier probes. arXiv:1610.01644, 2016.
- Belinkov Y. Probing classifiers: promises, shortcomings, and advances. *Computational Linguistics* 48(1):207-219, 2022.
- Bohanec M., Rajkovič V. Knowledge acquisition and explanation for multi-attribute decision making. *8th International Workshop on Expert Systems and their Applications*, Avignon, 1988, pp. 59-78. Data: UCI Machine Learning Repository, doi:10.24432/C5JP48.
- Costarelli A., Allen M., Field S. Meta-Models: an architecture for decoding LLM behaviors through interpreted embeddings and natural language. arXiv:2410.02472, 2024.
- Karvonen A. et al. Activation Oracles: training and evaluating LLMs as general-purpose activation explainers. *ICML 2026*, PMLR 306:56222-56258.
- Lapuschkin S. et al. Unmasking Clever Hans predictors and assessing what machines really learn. *Nature Communications* 10:1096, 2019.
- Pan A., Chen L., Steinhardt J. LatentQA: teaching LLMs to decode activations into natural language. *ICLR 2026*.
- Shaham T. R. et al. A multimodal automated interpretability agent. *ICML 2024*, PMLR 235:44293-44321.
- Sundararajan M., Taly A., Yan Q. Axiomatic attribution for deep networks. *ICML 2017*, PMLR 70:3319-3328.
- Zech J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine* 15(11):e1002683, 2018.

## Citation

```bibtex
@misc{nigmatov2026whybox,
  author = {Nigmatov, Ilyas},
  title  = {Causal interpretability of neural networks from their internal signals},
  year   = {2026},
  url    = {https://github.com/knbww/whybox}
}
```
