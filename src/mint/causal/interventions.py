"""Executed interventions on a target model B -- the causal ground truth.

Four distinct quantities, and keeping them apart is the point of the study:

`unit_effect[n, u]`      total causal effect of unit u at state n, measured by
                         mean-ablating u and reading the change in B's logit.
`unit_corr[u]`           correlation between u's activation and B's output over
                         the probe set: *predictive* evidence, no intervention.
`unit_ie[n, u, f]`       indirect effect of the human factor f through unit u:
                         patch u to the activation it takes under the executed
                         world counterfactual do(f := neutral), keep everything
                         else at state n.  This is what ties a human-vocabulary
                         explanation to a specific internal variable.
`factor_total[n, f]`     B's behavioural response to do(f := neutral) -- the
                         effect the per-unit explanation has to account for.

A unit with large |corr| and near-zero |effect| is a *decoy*: the interpreter
must learn to reject it, and no output-only or correlational method can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..domains import Domain, annotate
from ..models.target import TargetModel
from .layout import UnitLayout


@dataclass
class ProbeSummary:
    """Model-level statistics over a probe set (no focal state involved)."""

    act_mean: np.ndarray  # (U,)
    act_std: np.ndarray
    act_p10: np.ndarray
    act_p50: np.ndarray
    act_p90: np.ndarray
    frac_active: np.ndarray
    corr_out: np.ndarray  # corr(activation, logit)
    grad_absmean: np.ndarray
    gradact_mean: np.ndarray
    gradact_absmean: np.ndarray
    redundancy: np.ndarray  # max |corr| with another unit in the same site
    in_norm: np.ndarray
    out_norm: np.ndarray
    depth: np.ndarray
    kind_id: np.ndarray  # 0 = mlp neuron, 1 = attention head
    site_width: np.ndarray
    logit_mean: float
    logit_std: float


@dataclass
class CausalGroundTruth:
    base_logit: np.ndarray  # (N,)
    acts: np.ndarray  # (N, U)
    grads: np.ndarray  # (N, U) d logit / d unit
    first_order: np.ndarray  # (N, U) component-exact first-order ablation term
    first_order_cf: np.ndarray  # (N, U, F) the same, under each do(factor := neutral)
    unit_effect: np.ndarray  # (N, U)
    unit_ie: np.ndarray  # (N, U, F)
    factor_total: np.ndarray  # (N, F)
    decisive: np.ndarray  # (N,) human-annotated decisive factor
    decisive_margin: np.ndarray  # (N,)
    probe: ProbeSummary


KIND_ID = {"mlp_neuron": 0, "attn_head": 1}


def _forward_capture(model: TargetModel, layout: UnitLayout, x: torch.Tensor, want_grad: bool = True):
    logit, acts = model(x, capture=True)
    flat = layout.flatten(acts)
    if not want_grad:
        return logit.detach().numpy(), flat.detach().numpy(), None
    grads = torch.autograd.grad(logit.sum(), list(acts.values()), allow_unused=True)
    gd = {s: (g if g is not None else torch.zeros_like(acts[s])) for s, g in zip(acts.keys(), grads)}
    gflat = layout.flatten(gd, grad=True)
    return logit.detach().numpy(), flat.detach().numpy(), gflat.detach().numpy()


def _weight_norms(model: TargetModel, layout: UnitLayout) -> tuple[np.ndarray, np.ndarray]:
    """Incoming / outgoing weight norm per unit, in a name-free way."""
    in_n = np.ones(layout.n_units, dtype=np.float64)
    out_n = np.ones(layout.n_units, dtype=np.float64)
    params = dict(model.named_parameters())
    # Matching by shape alone assigns layer 0's weights to every site of the same
    # width.  Ask the model which parameters belong to each site instead, and
    # fall back to shape matching only for sites that do not say.
    hints = getattr(model, "site_norms", lambda site: None)
    for s, w in zip(layout.sites, layout.widths):
        o = layout.offsets[s]
        told = hints(s)
        if told is not None:
            in_n[o:o + w] = told[0].detach().numpy()
            out_n[o:o + w] = told[1].detach().numpy()
            continue
        cand_in = [v for k, v in params.items()
                   if k.endswith("weight") and v.dim() == 2 and v.shape[0] == w]
        cand_out = [v for k, v in params.items()
                    if k.endswith("weight") and v.dim() == 2 and v.shape[1] == w]
        if cand_in:
            in_n[o:o + w] = cand_in[0].detach().norm(dim=1).numpy()
        if cand_out:
            out_n[o:o + w] = cand_out[-1].detach().norm(dim=0).numpy()
    return in_n, out_n


def probe_summary(model: TargetModel, layout: UnitLayout, x_probe: torch.Tensor) -> ProbeSummary:
    logit, acts, grads = _forward_capture(model, layout, x_probe)
    a = acts
    sd = a.std(0) + 1e-8
    lg_c = logit - logit.mean()
    corr = ((a - a.mean(0)) * lg_c[:, None]).mean(0) / (sd * (logit.std() + 1e-8))

    red = np.zeros(layout.n_units)
    az = (a - a.mean(0)) / sd
    for s, w in zip(layout.sites, layout.widths):
        o = layout.offsets[s]
        blk = az[:, o:o + w]
        c = np.abs(blk.T @ blk / len(blk))
        np.fill_diagonal(c, 0.0)
        red[o:o + w] = c.max(1) if w > 1 else 0.0

    depth = np.zeros(layout.n_units)
    kind = np.zeros(layout.n_units)
    width = np.zeros(layout.n_units)
    for u in model.units:
        f = layout.offsets[u.site] + u.idx
        depth[f], kind[f], width[f] = u.depth, KIND_ID[u.kind], u.site_width
    in_n, out_n = _weight_norms(model, layout)
    return ProbeSummary(
        act_mean=a.mean(0), act_std=a.std(0), act_p10=np.percentile(a, 10, axis=0),
        act_p50=np.percentile(a, 50, axis=0), act_p90=np.percentile(a, 90, axis=0),
        frac_active=(np.abs(a) > 1e-6).mean(0), corr_out=np.nan_to_num(corr),
        grad_absmean=np.abs(grads).mean(0), gradact_mean=(grads * a).mean(0),
        gradact_absmean=np.abs(grads * a).mean(0), redundancy=red,
        in_norm=in_n, out_norm=out_n, depth=depth, kind_id=kind, site_width=width,
        logit_mean=float(logit.mean()), logit_std=float(logit.std() + 1e-8),
    )


SiteValues = dict[str, torch.Tensor]  # per-site component-wise replacement values


def exact_first_order(model: TargetModel, layout: UnitLayout, x: torch.Tensor,
                      site_means: SiteValues) -> np.ndarray:
    """(N, U) first-order prediction of each unit's mean-ablation effect.

    Mean ablation is a *replacement*, so the first-order term is
    `sum_j g_j (c_j - a_j)` over the unit's components.  Summarising the unit
    first and writing `(mean_j a_j - c) * sum_j g_j` is equal only when the
    gradient and the activation are uncorrelated across components, which they
    are not in an attention head.
    """
    logit, acts = model(x, capture=True)
    tensors = [acts[s] for s in layout.sites]
    grads = torch.autograd.grad(logit.sum(), tensors, allow_unused=True)
    cols = []
    for s, a, g in zip(layout.sites, tensors, grads):
        if g is None:
            g = torch.zeros_like(a)
        term = g * (site_means[s].unsqueeze(0) - a)
        cols.append(term if term.dim() == 2 else term.sum(dim=tuple(range(2, term.dim()))))
    return torch.cat(cols, dim=1).detach().numpy()


@torch.no_grad()
def capture_sites(model: TargetModel, layout: UnitLayout, x: torch.Tensor) -> SiteValues:
    """The full activation tensor of every site, detached."""
    _, acts = model(x, capture=True)
    return {s: acts[s].detach() for s in layout.sites}


@torch.no_grad()
def probe_site_means(model: TargetModel, layout: UnitLayout,
                     x_probe: torch.Tensor) -> SiteValues:
    """Component-wise mean activation over the probe set: the ablation baseline.

    Averaging over the probe batch but *not* over a unit's components is what
    keeps mean ablation exact on a transformer.
    """
    return {s: v.mean(dim=0) for s, v in capture_sites(model, layout, x_probe).items()}


def _edits(layout: UnitLayout, mask: torch.Tensor, target) -> dict:
    if isinstance(target, dict):
        return layout.edits_from_sites(mask, target)
    return layout.edits(mask, torch.as_tensor(target, dtype=torch.float32))


@torch.no_grad()
def patch_effects(model: TargetModel, layout: UnitLayout, x: torch.Tensor,
                  target_vals, base_logit: np.ndarray) -> np.ndarray:
    """Delta logit from patching each unit individually.

    `target_vals` is either a per-site dict of component-wise tensors (the
    structure-preserving path) or an (N, U) array of scalar summaries.
    """
    n, u_n = x.shape[0], layout.n_units
    out = np.zeros((n, u_n), dtype=np.float64)
    mask = torch.zeros(n, u_n, dtype=torch.bool)
    for u in range(u_n):
        mask.zero_(); mask[:, u] = True
        out[:, u] = model.logits(x, edits=_edits(layout, mask, target_vals)).numpy() - base_logit
    return out


@torch.no_grad()
def subset_patch_effect(model: TargetModel, layout: UnitLayout, x: torch.Tensor,
                        target_vals, mask_np: np.ndarray,
                        base_logit: np.ndarray) -> np.ndarray:
    """Delta logit from patching a per-example *set* of units in one pass."""
    mask = torch.as_tensor(mask_np, dtype=torch.bool)
    return model.logits(x, edits=_edits(layout, mask, target_vals)).numpy() - base_logit


def ground_truth(model: TargetModel, domain: Domain, focal_raw: np.ndarray,
                 probe_raw: np.ndarray, patch_mode: str = "component") -> CausalGroundTruth:
    """Everything measurable about how B computes its answer on `focal_raw`.

    `patch_mode="component"` replaces a unit's whole activation tensor;
    `"scalar"` writes the unit's mean and broadcasts it, which is what the
    earlier runs did and which is degenerate on transformers (see
    `results/frozen/ERRATA.md`).
    """
    model.eval()
    layout = UnitLayout.of(model)
    x_probe = torch.as_tensor(domain.to_model_input(probe_raw))
    probe = probe_summary(model, layout, x_probe)

    x = torch.as_tensor(domain.to_model_input(focal_raw))
    base_logit, acts, grads = _forward_capture(model, layout, x)

    n, u_n = len(base_logit), layout.n_units
    site_means = probe_site_means(model, layout, x_probe)  # ablation baseline
    ablation_target = (site_means if patch_mode == "component"
                       else np.repeat(probe.act_mean[None, :], n, axis=0))
    unit_effect = patch_effects(model, layout, x, ablation_target, base_logit)
    first_order = exact_first_order(model, layout, x, site_means)

    n_f = domain.spec.n_factors
    unit_ie = np.zeros((n, u_n, n_f), dtype=np.float64)
    first_order_cf = np.zeros((n, u_n, n_f), dtype=np.float64)
    factor_total = np.zeros((n, n_f), dtype=np.float64)
    for f in range(n_f):
        raw_cf = domain.do_neutral(focal_raw, f)
        x_cf = torch.as_tensor(domain.to_model_input(raw_cf))
        with torch.no_grad():
            lg_cf, acts_cf = model(x_cf, capture=True)
        factor_total[:, f] = lg_cf.numpy() - base_logit
        # the analytic causal pattern under the same world counterfactual: needed
        # by the semantic layer, and computable without touching activations
        first_order_cf[:, :, f] = exact_first_order(model, layout, x_cf, site_means)
        target = ({s: acts_cf[s].detach() for s in layout.sites} if patch_mode == "component"
                  else layout.flatten(acts_cf).numpy())
        unit_ie[:, :, f] = patch_effects(model, layout, x, target, base_logit)

    ann = annotate(domain, focal_raw)
    return CausalGroundTruth(
        base_logit=base_logit, acts=acts, grads=grads, first_order=first_order,
        first_order_cf=first_order_cf, unit_effect=unit_effect,
        unit_ie=unit_ie, factor_total=factor_total, decisive=ann.factor_id,
        decisive_margin=ann.margin, probe=probe,
    )


@torch.no_grad()
def counterfactual_acts(model: TargetModel, domain: Domain, focal_raw: np.ndarray) -> np.ndarray:
    """(N, U, F) scalar unit summaries under each executed do(factor := neutral).

    Kept for descriptors and for reproducing earlier measurements; interventions
    should use `counterfactual_site_acts`.
    """
    layout = UnitLayout.of(model)
    n, n_f = focal_raw.shape[0], domain.spec.n_factors
    out = np.zeros((n, layout.n_units, n_f), dtype=np.float64)
    for f in range(n_f):
        x_cf = torch.as_tensor(domain.to_model_input(domain.do_neutral(focal_raw, f)))
        _, acts = model(x_cf, capture=True)
        out[:, :, f] = layout.flatten(acts).numpy()
    return out


@torch.no_grad()
def counterfactual_site_acts(model: TargetModel, domain: Domain, focal_raw: np.ndarray,
                             factor: int) -> SiteValues:
    """Component-wise activations under one executed do(factor := neutral)."""
    layout = UnitLayout.of(model)
    x_cf = torch.as_tensor(domain.to_model_input(domain.do_neutral(focal_raw, factor)))
    return capture_sites(model, layout, x_cf)
