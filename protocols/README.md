# Protocols

Every confirmatory run in this project had its protocol written and committed before the run:
the claim, the compared arms, the metrics, the family of comparisons declared in advance and the
rule that decides the result. This folder makes all of them public.

- **Runs of 13 to 15 September 2026** (tables 5 to 9 of the competition paper) had their protocols
  in separate documents. They are published here verbatim, byte for byte as committed:
  `PREREG_*.md`. The result files name them `docs/PREREG_*.md`, their place in the private
  repository.
- **Runs from 22 September 2026 on** keep the protocol in the module docstring of the run's script
  in [`../scripts/`](../scripts/). The script records its own sha256 in its result file and refuses
  to start if it differs from the committed version.
- **Earlier runs** (August and early September 2026) are in [`earlier/`](earlier/), with what came of
  each of them.

## How the order of events can be checked

The commit hashes and times below come from the author's private development repository; this
repository is an export without that history. In that history each protocol commit is an ancestor
of the commit at which its run was made, and the result file was committed after the run. Git takes
its times from the author's computer, so they show the order of events, not an independent
timestamp.

The files here are identical to the committed ones: the git blob hash listed for each file is what
`git rev-parse <commit>:docs/<file>` returns in the private repository, and `git hash-object` returns
the same for the file here. From 3 October 2026 the scripts are also in this public repository,
where GitHub records when each commit was pushed.

## Protocols behind the competition paper

Section and table numbers refer to the competition version of the paper. Section numbers inside
the script docstrings refer to a longer draft of the paper and differ from these.

### Separate protocol documents (September 2026)

| paper | run | protocol (blob) | frozen, UTC | amendment before the run | run made at | result, committed | read-out |
|---|---|---|---|---|---|---|---|
| 3.1, table 5 | `follows_b` | [`PREREG_FOLLOWS_B.md`](PREREG_FOLLOWS_B.md) (`18a3086547d8`) | `ace8a3a`, 2026-09-13 18:40:28 | `e4d710f`, 18:42:06: only states whose margin between the two leading causes exceeds 0.10 are scored; chosen from statistics of the correct answers, before any arm was trained | `31556cd`, 19:08:54 | [`follows_b.json`](../results/follows_b.json), 19:26:15 | CONFIRMED |
| 3.3, table 7 | `transfer_b` | [`PREREG_TRANSFER_B.md`](PREREG_TRANSFER_B.md) (`bbaa835ef08c`) | `8e10633`, 2026-09-13 19:48:38 | none | `8e10633` | [`transfer_b.json`](../results/transfer_b.json), 19:56:35 | CONFIRMED |
| not in the paper | `statement` (tabular model) | [`PREREG_STATEMENT.md`](PREREG_STATEMENT.md) (`fe7b0a7214c9`) | `c152718`, 2026-09-14 01:26:02 | none | `15a2c65`, 01:27:27 | [`statement.json`](../results/statement.json), 01:32:27 | CONFIRMED |
| 3.2, table 6 | `statement_lm` (language model) | the same file | `c152718` | `efb1fac`, 12:18:45: the same design on a language target | `c66d231`, 12:40:33 | [`statement_lm.json`](../results/statement_lm.json), 12:49:13 | CONFIRMED |
| 3.4, table 8 | `volume_study` | [`PREREG_VOLUME.md`](PREREG_VOLUME.md) (`ff3d81edaa64`) | `8698985`, 2026-09-15 12:51:49 | none | `8698985` | [`volume_study.json`](../results/volume_study.json), 14:23:55 | none of the four comparisons is significant (adjusted p 0.28 to 0.39) |
| 3.5, table 9 | `signals_study` | [`PREREG_SIGNALS.md`](PREREG_SIGNALS.md) (`f55ede8d5dc2`) | `f5ccae4`, 2026-09-15 14:23:55 | none | `f5ccae4` | [`signals_study.json`](../results/signals_study.json), 15:22:32 | dynamic and snapshot signals together: higher on all states (adjusted p 0.026); each kind alone: no significant change |

- `PREREG_STATEMENT.md` is published in its amended form; Amendment 1 is a section at its end. The
  tabular run was made before the amendment.
- `follows_b.json` was produced from a modified working tree; a byte-identical rerun is documented
  in [`../results/frozen/ERRATA.md`](../results/frozen/ERRATA.md) §12. These scripts overwrite their
  output.
- The twelve held-out models of these runs come from four independent initialisations at three
  strengths of the decoy feature, while the p-values assume twelve independent units (ERRATA §11).

### Protocols in script docstrings (from 22 September 2026)

| paper | script | protocol commit (private), UTC | public commit | run started, UTC | the public script is the one that ran |
|---|---|---|---|---|---|
| 3.7 (main result on twelve independent initialisations) | [`seeds12_all.py`](../scripts/seeds12_all.py) | `3f110a9`, 2026-09-22 13:37:17 | none (before this repository) | 13:37:25 | yes (sha256 `bf6ddb1e30b1`) |
| not in the paper | [`typical_control.py`](../scripts/typical_control.py) | `738120e`, 2026-10-03 06:14:08 | none (before this repository) | 06:14:09 | yes (`b9a70576accb`) |
| 3.5 (dynamic signals on twelve initialisations, 4 to 64 segments) | [`dynamics_deep.py`](../scripts/dynamics_deep.py) | `4caed14`, 2026-10-03 06:28:09 | none (before this repository) | 06:28:10 | yes (`28f75765cdca`) |
| 3.7 (nine pretrained language models) | [`sentences_tricks.py`](../scripts/sentences_tricks.py) | `558f7a8`, 2026-10-03 07:31:54; Amendment 1 `7884735`, 10:45:46, which only added the `--load-in-4bit` and `--steps` options | `0c36794` (10:26:52, before Amendment 1); `5103fe9` (Amendment 1, 10:45:58) | 07:31:54 (GPT-2 and Qwen2.5-0.5B, locally) | the runs on GPT-2, Qwen2.5-0.5B, GPT-2 large, Qwen2.5-1.5B and Qwen2.5-3B used the version before Amendment 1 (sha256 `1282e82a4e89`, the file of public commit `0c36794`); the later runs used the current file (`d82303bee41f`) |
| 3.7 (Car Evaluation) | [`car_study.py`](../scripts/car_study.py) | `ec33e73`, 2026-10-03 12:20:06 | `8f95053` (with its result, after the run) | 12:20:14 | yes (`9bb233ca227e`) |
| table 5 and 3.7 (first-order pointer) | [`pointer_control.py`](../scripts/pointer_control.py) | `ca55ce5`, 2026-10-04 11:24:49 | `4e3a4a5` (11:25:06) | 11:32:20 | yes (`86ba58ec7434`) |
| 3.7 (pointer on the language models) | [`pointer_sentences.py`](../scripts/pointer_sentences.py) | `ca55ce5`, 2026-10-04 11:24:49 | `4e3a4a5` (11:25:06) | 11:25:01 (GPT-2 and Qwen2.5-0.5B, locally); the Colab runs later | yes (`c327d27e46e6`) |

The Colab runs record the public commit they ran from.

### Diagnostics without a protocol

These are reported as diagnostics, outside any protocol:

- [`lm_disagree_split.json`](../results/lm_disagree_split.json): the split of the language model's
  departure states into those where the world names a cause and those where it does not (paper 3.2).
  It reproduces `statement_lm.json` exactly before splitting.
- [`ablation_streams.json`](../results/ablation_streams.json): removing the signals of internal
  neurons, or the global signals, in three worlds (paper 3.7).
- `sentences_tricks_colab_qwen3_*_steps64.json`: the two Qwen3 models with 64 path segments
  instead of 16; Amendment 1 allows other segment counts only as a diagnostic.
- The post-hoc analyses inside `pointer_control.json` (`post_hoc_car`, the bootstrap intervals);
  the script marks them as post hoc.

## Earlier protocols

| protocol (blob) | frozen, UTC | amendment | run | result | what came of it |
|---|---|---|---|---|---|
| [`earlier/PREREG.md`](earlier/PREREG.md) (`c339f34a2c3b`) | `edbc82a`, 2026-08-21 04:13:01 | `e0bf920`, 06:29:23, written after the first run, before the second one on fresh targets | `rerun_core`, `rerun_core2` | [`rerun_core2.json`](../results/frozen/rerun_core2.json) | gate FAIL: the corrected analytic first-order term beats the learned interpreter on the language cell |
| [`earlier/PREREG_SEMANTIC.md`](earlier/PREREG_SEMANTIC.md) (`124e8f68010d`) | `14d3a9c`, 2026-08-22 14:02:29 | none | `semantic_confirm` | [`semantic_confirm.json`](../results/frozen/semantic_confirm.json) | gate passes, with the caveats in [`../results/frozen/README.md`](../results/frozen/README.md) |
| [`earlier/PREREG_STRUCTURE.md`](earlier/PREREG_STRUCTURE.md) (`c383d0227385`) | `e97259b`, 2026-08-24 19:32:36 | none | `structure_confirm` | [`structure_confirm.json`](../results/frozen/structure_confirm.json) | reading the internals adds nothing |
| [`earlier/PREREG_HYBRID.md`](earlier/PREREG_HYBRID.md) (`69888bd0f8b6`) | `dcc5466`, 2026-08-25 07:45:34 | none | `hybrid_confirm` | [`hybrid_confirm.json`](../results/frozen/hybrid_confirm.json) | primary comparison FAIL; the result was withdrawn (see ERRATA) |
| [`earlier/PREREG_WIDE.md`](earlier/PREREG_WIDE.md) (`f4ca1dfaaad0`) | `8021516`, 2026-09-08 13:17:33 | none | no committed script or result | none | superseded on 2026-09-13, when the correct answer was redefined as the cause the model itself relies on |

Some protocols mention other working notes of the private repository (`docs/RESULTS.md`,
`docs/INTERFACE.md`, `docs/FINDINGS.md`); those notes are not published. Two protocols carry the
authorship note they were committed with. A draft that was never frozen is not included.

The follow-up study on addresses of causes inside target models is not part of the competition
paper. The protocol of its stage R.0 is embedded in [`../colab/R0_pythia.ipynb`](../colab/R0_pythia.ipynb).
