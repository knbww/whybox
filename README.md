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
reads pretrained language models from GPT-2 to Qwen3-14B with the saved interpreter
(choose a GPU runtime).

## Why this matters

A model can be accurate for the wrong reason. A network trained to detect pneumonia on
chest X-rays learned to recognise which hospital a scan came from (Zech et al., 2018);
an image classifier recognised horses by a copyright tag (Lapuschkin et al., 2019).
Accuracy alone does not reveal this: one has to know what a decision rests on.
Attribution methods answer that per model and per decision, as an importance map.
Models that read other models describe states or behaviours in words, but do not check
the cause of a decision on the model itself. This project asks whether one interpreter,
trained once, can name the cause of a decision for models it has never seen, and whether
the skill transfers across models and tasks.

## The hypothesis

> If one AI model is trained to understand the internal causal state of another model
> from universal ML signals, it can give a sufficiently accurate, human-understandable
> causal interpretation that transfers between different target models and tasks.

Its parts, each judged on its own: (1) one trained interpreter; (2) it reads universal
signals of B; (3) it names the cause B itself relies on, not the world's; (4) accurately
enough: above chance, a constant answer and a predictor of the world; (5) in human terms;
(6) across target models and tasks it was not trained on. Each result below says which
parts it supports, leaves untested or contradicts.

## Results

### Simulated worlds, twelve independent held-out models

A tactical game world with seven causes (chance 14.3%). Protocol committed before the run:
[`scripts/seeds12_all.py`](scripts/seeds12_all.py),
[`scripts/typical_control.py`](scripts/typical_control.py).

| states | interpreter | constant | other reference |
|---|---|---|---|
| all | **73.8%** | 63.2% | perfect predictor of the world 87.7% |
| B departs from the world (12% of states) | **59.3%** | 28.1% | perfect predictor of the world **0%** |
| B departs from what 36 other models rely on (12%) | **54.9%** | 28.0% | predictor of typical models **0%**, inputs-only classifier 20.6% |

All primary comparisons hold after Holm correction (adjusted p from 0.003 to 0.012; the
smallest attainable with twelve units is 0.0005). This supports: one trained model (1), reading B's signals (2),
naming B's own cause rather than the world's or the typical model's (3), accuracy above
chance and a constant (4).
**Caveat:** on all states, predictors that never read B are more accurate (typical models
88.4%, inputs only 86.7%), because models mostly rely on the same thing. The pooled
accuracy alone does not show that B is read; the cells where B departs do.

**Dynamic signals** ([`scripts/dynamics_deep.py`](scripts/dynamics_deep.py)): integrated
gradients along the path to the neutral state raise accuracy to 78.5% (+4.7 points,
p = 0.005); on these small models the number of path segments (4 to 64) makes no
detectable difference.

### A real dataset: UCI Car Evaluation

1728 cars, six attributes, the published four-level acceptability, used as published.
Causes are the concepts of the published decision hierarchy: PRICE, COMFORT, SAFETY
(chance 33.3%). Twelve independent held-out initialisations, 14 comparisons under one
Holm correction, protocol committed before the run:
[`scripts/car_study.py`](scripts/car_study.py), result
[`results/car_study.json`](results/car_study.json).

| question | interpreter | reference | read-out |
|---|---|---|---|
| names the cause B relies on | **68.5%** | constant 38.7% | supported |
| where B departs from the published rule (0.85% of states) | **47.9%** | constant 17.0%, world 0% | supported |
| where B departs from what other models rely on (6.8%) | 39.0% | constant **48.9%** | **not supported** |
| trained only on the simulation, reads Car models | 60.3% (all states) | constant 38.7% | not supported: 37.3% vs 48.9% in the cell above |
| trained only on Car, reads simulation models | 61.4% (all states) | constant 60.1% | not supported on all states; in the hard cells 53.5% and 51.5% vs about 25% |
| dynamic signals | **85.6%** vs 68.5% | | better |

On real data the main reading holds, but the strict check ("this model, not models in
general") does not reproduce: where a Car model departs from the others, its top two
causes are almost tied, and the interpreter does not resolve such near-ties.

### Pretrained language models, full sentences with a trick

The interpreter was trained only on toy language models of about 40 thousand parameters and
is applied without further training. B chooses between *is* and *are* after sentences with
a relative clause, where a second noun pulls the verb's number the wrong way:
"The quiet nurse that the critics like ...". Causes: the number of the subject, the number
of the distracting noun, the adjective; a long template has seven causes. The floor is a
random choice among the causes whose suppression changes B at all. Protocol:
[`scripts/sentences_tricks.py`](scripts/sentences_tricks.py).

| model | precision | object clause | subject clause | long template |
|---|---|---|---|---|
| GPT-2 (124M) | fp32 | **1.000** / 0.562 | **0.928** / 0.548 | **0.886** / 0.301 |
| GPT-2 large (774M) | fp32 | **0.961** / 0.550 | **0.866** / 0.552 | **0.652** / 0.309 |
| GPT-2 XL (1.5B) | fp32 | **0.969** / 0.534 | **0.906** / 0.551 | **0.736** / 0.307 |
| Qwen2.5-0.5B | fp32 | **0.977** / 0.562 | **0.919** / 0.579 | **0.792** / 0.308 |
| Qwen2.5-1.5B | fp32 | **0.962** / 0.582 | **0.883** / 0.567 | **0.790** / 0.313 |
| Qwen2.5-3B | fp16 | **0.954** / 0.571 | **0.915** / 0.581 | **0.818** / 0.308 |
| Qwen2.5-7B | 4-bit | **0.944** / 0.578 | **0.882** / 0.568 | **0.808** / 0.326 |
| Qwen3-8B | 4-bit | **0.905** / 0.603 | **0.816** / 0.608 | **0.735** / 0.346 |
| Qwen3-14B | 4-bit | **0.867** / 0.562 | **0.704** / 0.600 | **0.726** / 0.316 |

Interpreter / floor; every cell is above the floor after Holm correction. This supports
parts 1, 2, 4, 5 and 6 of the hypothesis on models up to 14 billion parameters. Where a
model departs from grammar (81 sentences over the nine models), the interpreter names the
model's own cause in 74% of them, against 39% for the floor and 0% for a predictor of
grammar; this pooled figure is descriptive, not a pre-registered test.

Reading weakens for the largest models. A diagnostic shows why: the 16-segment path is too
coarse for them. With 64 segments the completeness error falls four to eight times and
Qwen3-8B reads at 0.957 / 0.917 / 0.827, Qwen3-14B at 0.948 / 0.917 / 0.835
(diagnostic runs, outside the protocol's read-out).

### Earlier results, with a qualification

Before the twelve-initialisation design, held-out models came from four independent seeds
at three decoy strengths, so p-values that assume twelve units overstate the resolution;
every effect below is positive in all four seed groups. A transformer language model in a
toy logic world: 71.4% (chance 25%). Transfer between unrelated simulated worlds: trained
only on medical models, the interpreter reads tactical ones at 70.5% (72.7% when trained on
them); in the other direction 54.1%.

## Limitations

- The causes and the inputs through which they act are listed in advance; the interpreter
  points at inputs and the name comes from that list. Two causes acting through the same
  input are indistinguishable, and a cause outside the list cannot be named.
- One cause per decision. States without a clear main cause (a fifth to a third of states,
  depending on the task) are excluded.
- The interpreter reads gradient-based estimates of influence, not raw activations or
  weights. Direction and size of the effect are measured on B, not predicted by A.
- "This model, not models in general" holds on simulated worlds and not on Car Evaluation.
- On pretrained language models the departures from grammar are too few per model for a
  separate test.
- Path signals need more segments for large models; a future protocol should fix the rule
  for choosing the number of segments in advance.

## How the work is checked

Every confirmatory run has its protocol (claim, compared arms, metrics, the family of
comparisons and the read-out rule) in the docstring of its script, committed before the
run. The scripts refuse to run unless they equal the committed version and refuse to
overwrite their result. Results in [`results/`](results) carry the commit, configuration
and seeds. Statistics: exact sign-flip tests paired over independent initialisations,
Holm correction over the declared family.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" transformers
.venv/bin/python -m pytest -q
```

- **Pretrained language models:** the Colab notebook above, or
  `PYTHONPATH=.:src:scripts python scripts/sentences_tricks.py --models gpt2 --out my.json`.
  The interpreter weights are in [`results/sentences_tricks_interpreter.pt`](results/sentences_tricks_interpreter.pt).
- **Simulated worlds and Car Evaluation:** these read cached target populations
  (`results/cache/`, about 5 GB, not in the repository). Builder scripts in `scripts/`
  recreate them (for example `fresh_build.py`, `volume_build.py`, `car_build.py`), and every
  consumer checks the sha256 recorded in the manifests under `results/`; a rebuild with other
  library versions may differ bit for bit, and then the check stops the run.
  A finished run refuses to overwrite its result; move the result file away to run it again.

## Data and models

- `data/uci/`: Car Evaluation (Bohanec & Rajkovič, 1988), Balance Scale (Siegler, 1976) and
  Tic-Tac-Toe Endgame (Aha, 1991) from the UCI Machine Learning Repository, CC BY 4.0, with
  their original description files.
- Pretrained models (GPT-2, Qwen2.5, Qwen3) are downloaded from Hugging Face under their own
  licenses and are not redistributed here.

## Related work

Costarelli, Allen & Field (2024, arXiv:2410.02472) trained meta-models that read another
model's activations, answer in natural language and transfer across model families. What
is specific here: the interpreter names the cause of a particular decision, every answer is
checked by an intervention on the model, and the correct answer is what the model itself
relies on rather than what drives the world.

## License

Code: [MIT](LICENSE). Text and figures: [CC BY 4.0](LICENSE-CC-BY-4.0.txt). The UCI data keep
their own license (CC BY 4.0).

## Use of AI tools

Most of the code and the experiment scripts were written with an AI coding assistant
(Claude, Anthropic) under the author's direction. The research question, the hypothesis
and the design decisions are the author's.

## Citation

```bibtex
@misc{nigmatov2026whybox,
  author = {Nigmatov, Ilyas},
  title  = {Causal interpretability of neural networks from their internal signals},
  year   = {2026},
  url    = {https://github.com/knbww/whybox}
}
```
