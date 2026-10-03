# whybox

Code for a student research project on causal interpretability: a trained interpreter
reads the internal signals of another neural network and names the cause of its
decision, and every answer is checked by an intervention on that network.

A full description is in preparation.

**Try it on Colab:** open
[`colab/sentences_tricks_colab.ipynb`](https://colab.research.google.com/github/knbww/whybox/blob/main/colab/sentences_tricks_colab.ipynb)
(GPT-2 and Qwen on sentences with relative clauses; choose a GPU runtime).

**Run locally:**

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" transformers
.venv/bin/python -m pytest -q
```
