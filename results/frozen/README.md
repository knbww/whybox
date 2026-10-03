# Frozen results

> **Read [`ERRATA.md`](ERRATA.md) before citing anything in this directory.** For what currently stands, see [`../../docs/FINDINGS.md`](../../docs/FINDINGS.md).

Snapshot of the causal-transfer stage, kept immutable so later work cannot
quietly restate it. Regenerate with the configs named below; do not overwrite.

| file | config | wall | what it settled |
|---|---|---|---|
| `pilot.json` | `configs/pilot.yaml` | 1690 s | first pilot: cross-ontology transfer works, cross-architecture appeared to fail |
| `llm_transfer.json` | `configs/llm_transfer.yaml` | 12287 s | the apparent architecture failure was source diversity; localisation transfers to a real LM at 86% of the in-cell ceiling |
| `explanation.json` | `configs/explanation.yaml` | 2369 s | naming does not transfer from tabular sources; 1-NN retrieval beats every learned arm; the end-to-end score is invalid on shared-circuit models |
| `semantic.json` | `configs/semantic.yaml` | 6713 s | with an LM source, naming transfers to unseen LMs of the same task and beats retrieval (0.939 vs 0.891); beyond that it only matches retrieval. **Caveat: the `A_output_only` naming rows in this file are not internals-blind** — see docs/RESULTS.md |

| `rerun_core2.json` | `configs/rerun_core2.yaml` | 6413 s | **confirmatory.** Corrected instrument, fresh targets, 12/12 split, 3 interpreter seeds. Gate returns FAIL: the corrected analytic first-order term beats the learned interpreter on the language cell (-0.226, adj. p 0.016); the interpreter wins on both MLP cells. Written to `results/pilot.json` by a config slip and renamed; see the `note` field. |

| `semantic_confirm.json` | `scripts/semantic_confirm.py` | 5471 s | **confirmatory.** Localisation removed from the learned model; input is the first-order causal pattern. Gate PASSES: naming transfers from `agreement/lm` to `polarity/lm`, +0.179 over the internals-blind control (adj. p 0.006). Negative control (`skirmish/mlp`) clean. Reaches 59% of the in-cell ceiling; a parameter-free assisted lookup still beats it. |

| `structure_confirm.json` | `scripts/structure_confirm.py` | 3162 s | **confirmatory.** Reading the *kind* of causal structure, the one task where the analytic baseline was shown to fail first. Internals add nothing (±0.01, adj. p 0.650) and do not beat random alone; the transferable signal is a learned classification of the per-position first-order pattern, beating the heuristic by +0.11 in one direction (adj. p 0.016) and not in the other. |

| `hybrid_confirm.json` | `scripts/hybrid_confirm.py` | see file | **confirmatory.** Primary (whole-statement exact) FAILS: hybrid vs aligned instrument is +0.003 in both directions, adj. p 1.000. Co-primary (executed validity) held at +0.173 and +0.430, adj. p 0.004. **WITHDRAWN** — the comparator was beaten by a zero-information constant, a claim proposing no intervention scored 0.665 on the co-primary, and the learned arms were trained on labels disagreeing with the evaluation truth on ~37% of states. Registered verdict: FAIL. See ERRATA and `docs/FINDINGS.md`. |

| `scaling_study.json`, `scaling_readout.json` | `scripts/scaling_study.py`, `scripts/scaling_report.py` | see files | **scaling, exploratory with a frozen read-out.** 96 cells: four capacity rungs (43k-3.20M), four levels of causal experience (3-24 fitted targets), both directions, three seeds, the same twelve held-out targets throughout. Capacity axis: 0 of 10 tests pass, structure *falls* with capacity in one direction (adj. p 0.022). Experience axis: executed validity +0.182 and +0.241, both adj. p 0.0098. Declared verdict MIXED/ASYMMETRIC on a 0.003 monotonicity dip. **Provisional** — inherits the label defect and the broken comparator of `hybrid_confirm`; see ERRATA. |

Narrative and caveats: `docs/RESULTS.md`. Nothing in this directory is an input
to any later run.
