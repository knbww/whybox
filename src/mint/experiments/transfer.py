"""The transfer experiment.

Protocol
--------
1. Train a heterogeneous population of target models B in every domain.
2. Measure the causal ground truth on each: per-unit ablation effects, per-unit
   indirect effects of the human-annotated decisive factor, decoy status.
3. Split each population into *fit* and *held-out* models.
4. Train interpreter A only on the **source** domain's fit models.
5. Evaluate zero-shot on the **target** domains' held-out models, against
   correlational / gradient / output-only baselines and an in-domain ceiling.

Arms
----
A_full            interpreter with procedure + vocabulary supervision
A_procedure_only  same, lambda_vocab = 0 (does human vocabulary help or hurt?)
A_output_only     same capacity and training, internals blanked out
A_ceiling_<dom>   interpreter trained on the target domain itself (upper bound)
baselines         correlation, activation, gradient, grad x act, first-order, ...
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..domains import get_domain
from ..eval.explanation import carrier_selectivity, evaluate_explanations, naming_baselines
from ..eval.harness import baseline_scores, evaluate, table
from ..eval.stats import comparison_table, min_attainable_p, run_comparisons
from ..interpreter.baselines import ASSISTED_BASELINES, BASELINES
from ..interpreter.dataset import Bundle, build_bundles, to_problem_set
from ..interpreter.model import FEATURE_GROUPS, INTERNAL_FEATURES
from ..interpreter.train import TrainConfig, predict, train_interpreter
from ..models.train_target import DEFAULT_FAMILY, population


CACHE_VERSION = 5  # v5: adds the analytic causal pattern under each world counterfactual


@dataclass
class ExperimentConfig:
    run_name: str = "pilot"
    # An experiment *cell* is a domain paired with an architecture family.  Two
    # cells over the same domain isolate the architecture axis; two cells over
    # the same family isolate the ontology axis.
    cells: tuple[dict, ...] = ()
    source_cells: tuple[str, ...] = ()
    # Legacy convenience: cells are derived from these when `cells` is empty.
    source_domains: tuple[str, ...] = ("skirmish",)
    target_domains: tuple[str, ...] = ("clinic", "seqworld")
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    spurious: tuple[float, ...] = (0.0, 1.0, 2.0)
    target_n_train: int = 6000
    target_epochs: int = 60
    target_weight_decay: float = 1e-4
    target_label_smoothing: float = 0.0
    n_focal: int = 160
    n_probe: int = 512
    holdout_models: int = 4
    interpreter: dict = field(default_factory=dict)
    feature_ablations: bool = True
    # Drop states whose decisive factor barely moves B from the mediation loss.
    # 0.0 reproduces the first runs; 0.25 matches the evaluation filter.
    mediation_min_effect: float = 0.0
    # Extra interpreters fitted on a different set of source cells, e.g. to ask
    # whether architectural diversity in the source is what unlocks transfer.
    # Each entry: {name, cells: [...]}.  Training sets are subsampled to the
    # size of the primary one so the comparison is data-matched.
    extra_source_arms: tuple[dict, ...] = ()
    # Which cells get an in-cell ceiling interpreter; empty means every non-source
    # cell.  Ceilings are the most expensive arms, so this is the cost dial.
    ceiling_cells: tuple[str, ...] = ()
    explanation_eval: bool = True
    # The closed-vocabulary head is deprecated; its ablation arm costs a full
    # interpreter and answers a question the open vocabulary replaced.
    legacy_vocab_arm: bool = False
    # Factors removed from the candidate set while training, restored at
    # evaluation: the test of naming a cause never seen during training.
    holdout_factors: dict = field(default_factory=dict)
    # Declared before the run and corrected as one family; see docs/PREREG.md.
    primary_comparisons: tuple[dict, ...] = ()
    alpha: float = 0.05
    # "strict": the interpreter sees neither the candidate locus profile nor the
    # retrieval posterior, both of which are executed-intervention measurements
    # on the target model's reference states.  "assisted": it sees both, and so
    # do the baselines -- `locus_retrieved` in particular, which has no learned
    # parameters and which the strict condition exists to keep out.
    condition: str = "strict"
    # Several interpreter seeds: a single seed cannot separate an arm difference
    # from initialisation noise.  Metrics are averaged per target model across
    # seeds, so the bootstrap unit stays the model.
    interpreter_seeds: tuple[int, ...] = (0,)
    config_path: str = ""
    data_seed: int = 0
    cache_dir: str = "results/cache"
    out_dir: str = "results"

    def resolve(self) -> tuple[list[tuple[str, str]], list[str]]:
        """-> ([(domain, family), ...], [source cell name, ...])"""
        if self.cells:
            cells = [(c["domain"], c.get("family") or DEFAULT_FAMILY[get_domain(c["domain"]).spec.kind])
                     for c in self.cells]
            sources = list(self.source_cells) or [f"{d}/{f}" for d, f in cells[:1]]
        else:
            names = list(self.source_domains) + [d for d in self.target_domains
                                                 if d not in self.source_domains]
            cells = [(d, DEFAULT_FAMILY[get_domain(d).spec.kind]) for d in names]
            sources = [f"{d}/{DEFAULT_FAMILY[get_domain(d).spec.kind]}" for d in self.source_domains]
        seen, ordered = set(), []
        for c in cells:
            if c not in seen:
                seen.add(c); ordered.append(c)
        return ordered, sources


def provenance(cfg: ExperimentConfig) -> dict:
    """Everything needed to say which code and which inputs produced a result."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # pragma: no cover - git absent
            return ""
    import torch as _t
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "config_path": cfg.config_path,
        "condition": cfg.condition,
        "target_seeds": list(cfg.seeds),
        "target_spurious": list(cfg.spurious),
        "interpreter_seeds": list(cfg.interpreter_seeds),
        "cache_version": CACHE_VERSION,
        "torch": _t.__version__,
    }


def cache_key(domain_name: str, family: str, cfg: ExperimentConfig) -> str:
    """Bundles are only reusable if every knob that shaped them is unchanged."""
    payload = json.dumps({
        "v": CACHE_VERSION, "domain": domain_name, "family": family,
        "seeds": list(cfg.seeds), "spurious": list(cfg.spurious),
        "n_train": cfg.target_n_train, "epochs": cfg.target_epochs,
        "wd": cfg.target_weight_decay, "ls": cfg.target_label_smoothing,
        "n_focal": cfg.n_focal, "n_probe": cfg.n_probe, "data_seed": cfg.data_seed,
    }, sort_keys=True)
    return f"{domain_name}-{family}-{hashlib.sha1(payload.encode()).hexdigest()[:10]}"


def build_or_load(domain_name: str, family: str, cfg: ExperimentConfig,
                  refresh: bool = False) -> list[Bundle]:
    cache = Path(cfg.cache_dir) / f"{cache_key(domain_name, family, cfg)}.pt"
    if cache.exists() and not refresh:
        print(f"[{domain_name}/{family}] loading cached bundles from {cache}")
        return torch.load(cache, weights_only=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    domain = get_domain(domain_name)
    n_train = cfg.target_n_train
    if domain.spec.kind in ("sequence", "language"):
        n_train = max(n_train, 8000)
    cfgs = population(domain, seeds=cfg.seeds, spurious=cfg.spurious, epochs=cfg.target_epochs,
                      n_train=n_train, weight_decay=cfg.target_weight_decay,
                      label_smoothing=cfg.target_label_smoothing, family=family)
    print(f"[{domain_name}/{family}] training {len(cfgs)} target models")
    bundles = build_bundles(domain_name, cfgs, n_focal=cfg.n_focal, n_probe=cfg.n_probe,
                            seed=cfg.data_seed)
    torch.save(bundles, cache)
    return bundles


def split_bundles(bundles: list[Bundle], holdout: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Held-out models are *models the interpreter has never read*."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(bundles))
    return sorted(idx[holdout:].tolist()), sorted(idx[:holdout].tolist())


def run(cfg: ExperimentConfig, refresh: bool = False, run_executed: bool = True) -> dict:
    t0 = time.time()
    cell_specs, source_cells = cfg.resolve()
    cell_names = [f"{d}/{f}" for d, f in cell_specs]
    domains = sorted({d for d, _ in cell_specs})
    print(f"cells: {cell_names}  |  fitting on {source_cells}")

    per_cell: dict[str, list[Bundle]] = {}
    all_bundles: list[Bundle] = []
    offsets: dict[str, int] = {}
    for (d, f), name in zip(cell_specs, cell_names):
        per_cell[name] = build_or_load(d, f, cfg, refresh)
        offsets[name] = len(all_bundles)
        all_bundles.extend(per_cell[name])
    holdout = {k: tuple(v) for k, v in (cfg.holdout_factors or {}).items()}
    ps = to_problem_set(all_bundles, domains=domains, cells=cell_names,
                        mediation_min_effect=cfg.mediation_min_effect,
                        holdout_factors=holdout)
    if holdout:
        print(f"held out of training: {holdout}")
    print(f"problem set: {len(ps)} (model, state) problems, "
          f"U_max={ps.unit_x.shape[1]}, D={ps.unit_x.shape[2]}")

    fit_ids: dict[str, list[int]] = {}
    held_ids: dict[str, list[int]] = {}
    for name in cell_names:
        if cfg.holdout_models >= len(per_cell[name]):
            raise ValueError(f"holdout_models={cfg.holdout_models} leaves no models to fit on "
                             f"in {name!r} (population has {len(per_cell[name])})")
        f_, h_ = split_bundles(per_cell[name], cfg.holdout_models, seed=cfg.data_seed)
        fit_ids[name] = [i + offsets[name] for i in f_]
        held_ids[name] = [i + offsets[name] for i in h_]

    bid = ps.bundle_id.numpy()
    rows_of = lambda ids: torch.as_tensor(np.nonzero(np.isin(bid, ids))[0])
    factor_heads = {d: get_domain(d).spec.n_factors for d in domains}
    src_domains = sorted({c.split("/")[0] for c in source_cells})

    train_ps = ps.subset(rows_of([i for c in source_cells for i in fit_ids[c]]))
    if len(train_ps) == 0:
        raise ValueError("empty interpreter training set")
    eval_sets = {c: ps.subset(rows_of(held_ids[c])) for c in cell_names}
    print(f"train on {len(train_ps)} problems; eval on " +
          ", ".join(f"{c}:{len(v)}" for c, v in eval_sets.items()))

    assisted = cfg.condition == "assisted"
    interp_cfg = dict(cfg.interpreter)
    if not assisted:
        interp_cfg.update(drop_locus=True, use_retrieval=False)
    base_cfg = TrainConfig(**interp_cfg)
    print(f"condition: {cfg.condition} "
          f"(locus/retrieval {'visible to every method' if assisted else 'withheld'})")
    src_heads = {d: factor_heads[d] for d in src_domains}
    arms: dict[str, list] = {}
    seeds_i = list(cfg.interpreter_seeds)

    def fit(subset, tcfg_over: dict, heads: dict, verbose: bool = False):
        """One interpreter per seed."""
        return [train_interpreter(subset, TrainConfig(**{**interp_cfg, **tcfg_over, "seed": sd}),
                                  heads, verbose=verbose and sd == seeds_i[0])
                for sd in seeds_i]
    locus_free: set[str] = set()
    retrieval_free: set[str] = set()

    def heldout_mask(sub) -> np.ndarray | None:
        """Which problems have a decisive cause that training never saw."""
        if not holdout:
            return None
        return ~sub.train_factor_mask.numpy()[np.arange(len(sub)), sub.y_factor.numpy()]
    print("\n== A_full (procedure + vocabulary supervision, source cells only) ==")
    arms["A_full"] = fit(train_ps, {}, src_heads, verbose=True)
    if not assisted:
        locus_free.add("A_full"); retrieval_free.add("A_full")
    if cfg.legacy_vocab_arm:
        print("\n== A_procedure_only (no closed-vocabulary loss) ==")
        arms["A_procedure_only"] = fit(train_ps, {"lambda_vocab": 0.0}, src_heads)
    # A genuine internals-blind control has to close *three* channels, not one:
    # the unit descriptors, the candidate locus profile, and the retrieval
    # posterior -- the last two are computed from B's activations and bypass the
    # descriptor mask entirely.  Closing only the first leaves a naming head that
    # still reads the model.
    print("\n== A_output_only (no unit descriptors, no locus, no retrieval) ==")
    arms["A_output_only"] = fit(train_ps, {"drop_features": INTERNAL_FEATURES,
                                           "drop_locus": True, "use_retrieval": False}, src_heads)
    locus_free.add("A_output_only")
    retrieval_free.add("A_output_only")
    if cfg.feature_ablations:
        for group, feats in FEATURE_GROUPS.items():
            print(f"\n== A_no_{group} (that descriptor family blanked) ==")
            arms[f"A_no_{group}"] = fit(train_ps, {"drop_features": feats}, src_heads)
    for extra in cfg.extra_source_arms:
        cells_ = list(extra.get("cells") or source_cells)
        rows_ = rows_of([i for c in cells_ for i in fit_ids[c]])
        g = torch.Generator().manual_seed(cfg.data_seed)
        rows_ = rows_[torch.randperm(len(rows_), generator=g)[:len(train_ps)]]
        heads_ = {c.split("/")[0]: factor_heads[c.split("/")[0]] for c in cells_}
        tc = TrainConfig(**{**cfg.interpreter, **extra.get("overrides", {})})
        print(f"\n== {extra['name']} (fitted on {cells_}, {len(rows_)} problems) ==")
        arms[extra["name"]] = fit(ps.subset(rows_), extra.get("overrides", {}), heads_)
        if tc.drop_locus:
            locus_free.add(extra["name"])
        if not tc.use_retrieval:
            retrieval_free.add(extra["name"])
    ceilings = [c for c in (cfg.ceiling_cells or cell_names)
                if c not in source_cells and c in cell_names]
    for name in ceilings:
        print(f"\n== A_ceiling_{name} (trained in-cell: upper bound for transfer) ==")
        arms[f"A_ceiling_{name}"] = fit(ps.subset(rows_of(fit_ids[name])), {},
                                        {name.split("/")[0]: factor_heads[name.split("/")[0]]})
        if not assisted:
            locus_free.add(f"A_ceiling_{name}"); retrieval_free.add(f"A_ceiling_{name}")

    rows: list[dict] = []
    def preds(name, models, es):
        return [predict(m, es, drop_locus=name in locus_free,
                        use_retrieval=name not in retrieval_free) for m in models]

    for name, models in arms.items():
        for c, es in eval_sets.items():
            if name.startswith("A_ceiling_") and name[len("A_ceiling_"):] != c:
                continue
            rows += evaluate(es, all_bundles, preds(name, models, es), name,
                             run_executed=run_executed)
    suite = dict(BASELINES)
    if assisted:
        suite.update(ASSISTED_BASELINES)
    for bname, fn in suite.items():
        for c, es in eval_sets.items():
            rows += evaluate(es, all_bundles, baseline_scores(es, fn), bname,
                             run_executed=run_executed)
    for c, es in eval_sets.items():
        oracle = es.y_effect.numpy().copy()
        oracle[~es.mask.numpy()] = -1e9
        rows += evaluate(es, all_bundles, oracle, "oracle_effect", run_executed=run_executed)

    # The explanation task: name the cause, localise it, predict its size, and
    # have all three survive an executed intervention.
    exp_rows: list[dict] = []
    if cfg.explanation_eval:
        print("\nevaluating explanations")
        for name, model in arms.items():
            for c, es in eval_sets.items():
                if name.startswith("A_ceiling_") and name[len("A_ceiling_"):] != c:
                    continue
                pr = preds(name, model, es)
                exp_rows += evaluate_explanations(
                    es, all_bundles, [q["name"].argmax(1) for q in pr], name,
                    carrier=[q["named_carrier"] for q in pr],
                    predicted_effect=[q["factor_effect"] for q in pr],
                    run_executed=run_executed, heldout=heldout_mask(es))
        ref_name = "A_multiarch" if "A_multiarch" in arms else "A_full"
        for c, es in eval_sets.items():
            # NOTE: naming baselines are scored with the interpreter's carriers,
            # so only the *name* differs.  Stated here because it is not neutral.
            carrier = preds(ref_name, arms[ref_name], es)[0]["named_carrier"]
            for bname, chosen in naming_baselines(es, all_bundles).items():
                exp_rows += evaluate_explanations(es, all_bundles, chosen, bname,
                                                  carrier=carrier, run_executed=run_executed,
                                                  heldout=heldout_mask(es))
            exp_rows += evaluate_explanations(es, all_bundles, es.y_factor.numpy(),
                                              "oracle_name", carrier=carrier,
                                              run_executed=run_executed,
                                              heldout=heldout_mask(es))

    stats = run_comparisons(rows + exp_rows, list(cfg.primary_comparisons), cfg.alpha,
                            cells=cell_names) if cfg.primary_comparisons else []
    if stats:
        print(f"\ndeclared comparisons (Holm-Bonferroni over {len(stats)}, "
              f"smallest attainable p at {cfg.holdout_models} models = "
              f"{min_attainable_p(cfg.holdout_models):.4f})")
        print(comparison_table(stats))

    out = {
        "provenance": provenance(cfg),
        "comparisons": stats,
        "config": asdict(cfg),
        "cells": cell_names,
        "source_cells": source_cells,
        "condition": cfg.condition,
        "rows": rows,
        "explanation_rows": exp_rows,
        "targets": {c: [b.trained.meta() for b in per_cell[c]] for c in cell_names},
        "carrier_selectivity": {
            c: float(np.nanmean([carrier_selectivity(all_bundles[i], eval_sets[c].state_id.numpy()[
                eval_sets[c].bundle_id.numpy() == i]) for i in held_ids[c][:3]]))
            for c in cell_names} if run_executed else {},
        "splits": {c: {"fit": fit_ids[c], "held_out": held_ids[c]} for c in cell_names},
        "wall_seconds": time.time() - t0,
    }
    outdir = Path(cfg.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{cfg.run_name}.json").write_text(json.dumps(out, indent=2, default=float))
    torch.save({"arms": arms, "cells": cell_names, "source_cells": source_cells,
                "held_ids": held_ids}, outdir / f"{cfg.run_name}_arms.pt")
    print(f"\nwrote {outdir / (cfg.run_name + '.json')}  ({out['wall_seconds']:.0f}s)")
    return out


# The headline task: propose the internal variables that carry a human-named
# causal factor, then have that proposal survive an executed intervention.
LOCALISATION = ["mediation.spearman", "mediation.precision@3",
                "executed.sufficiency@3", "executed.comprehensiveness@3",
                "decoy_rejection"]
# Diagnostic task: rank units by their single-unit ablation effect.  Largely
# solvable analytically -- `first_order` is the control that proves it.
IMPORTANCE = ["causal.spearman", "causal.precision@3", "causal.top1_effect_ratio"]


EXPLANATION = ["naming.accuracy", "naming.balanced_accuracy",
               "naming.accuracy_unseen_factor", "effect.rho", "effect.sign_accuracy",
               "explanation.score@3"]


def report(res: dict) -> str:
    parts = []
    source = set(res.get("source_cells", []))
    for d in res.get("cells", []):
        rows = [r for r in res.get("explanation_rows", []) if r["cell"] == d]
        if not rows:
            continue
        src = "source" if d in source else "TRANSFER"
        parts.append(f"\n### {d}  [{src}]  explanation task: name the cause, "
                     f"localise it, survive the intervention")
        sel = res.get("carrier_selectivity", {}).get(d)
        if sel is not None and np.isfinite(sel):
            note = "patch test informative" if sel > 0.15 else "PATCH TEST BLIND (carriers bottleneck)"
            parts.append(f"    carrier selectivity {sel:.3f} -- {note}")
        parts.append(table(rows, EXPLANATION, sort_by="naming.balanced_accuracy"))
    for d in res.get("cells") or sorted({r["cell"] for r in res["rows"]}):
        rows = [r for r in res["rows"] if r["cell"] == d]
        if not rows:
            continue
        src = "source" if d in source else "TRANSFER"
        parts.append(f"\n### {d}  [{src}]  ({rows[0]['n_models']} held-out models)")
        parts.append("\n-- localisation: which internal variables mediate the "
                     "human-named factor, validated by intervention --")
        parts.append(table(rows, LOCALISATION, sort_by="executed.sufficiency@3"))
        parts.append("\n-- diagnostic: single-unit ablation-effect ranking --")
        parts.append(table(rows, IMPORTANCE, sort_by="causal.spearman", show_ci=False))
    return "\n".join(parts)
