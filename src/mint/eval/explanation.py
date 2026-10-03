"""From a score over units to an explanation -- and then to a verdict.

An explanation here is a structured, falsifiable object:

    at this state, B's output is driven by *this* candidate cause,
    carried by *these* internal variables,
    and removing the cause would move the output by *this much*.

Every clause is checkable by executing an intervention on B, and this module
checks them.  The end-to-end number is the honest one: the counterfactual that
gets executed is the one the interpreter *named*, not the one the annotation
says is right, so a confidently wrong name is punished rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domains import get_domain
from ..interpreter.dataset import Bundle, ProblemSet
from .metrics import spearman
from .protocol import executed_protocol

# indices into a factor signature, see mint.encoding.factor_signature
SIG_MEAN_ABS_EFFECT = 1
SIG_BASE_RATE = 5


@dataclass
class Explanation:
    cell: str
    model_tag: str
    state_id: int
    named: int
    named_name: str
    correct: bool
    carriers: tuple[int, ...]
    predicted_effect: float
    measured_effect: float


def carrier_selectivity(bundle: Bundle, state_idx: np.ndarray, k: int = 3) -> float:
    """Can a patch test distinguish candidates in this model at all?

    Patch the *true* top-k carriers with each candidate's counterfactual in turn
    and take the spread of the achieved effect across candidates, relative to
    its size.  Near zero means the carriers are a bottleneck: any candidate's
    counterfactual moves the output by the same amount, so patch-based
    sufficiency is blind to which cause was named and must not be used to score
    naming in that cell.
    """
    import torch

    from ..causal.interventions import subset_patch_effect
    from ..causal.layout import UnitLayout
    from .protocol import gather_counterfactual

    gt, model = bundle.gt, bundle.trained.model
    domain = get_domain(bundle.domain)
    layout = UnitLayout.of(model)
    sel = np.asarray(state_idx)
    raw = bundle.focal_raw[sel]
    n, n_f = len(sel), gt.factor_total.shape[1]
    ie = np.abs(gt.unit_ie[sel][np.arange(n), :, gt.decisive[sel]])
    order = np.argsort(-ie, axis=1)[:, :k]
    mask = np.zeros((n, layout.n_units), dtype=bool)
    mask[np.repeat(np.arange(n)[:, None], k, axis=1), order] = True
    x = torch.as_tensor(domain.to_model_input(raw))
    tot = gt.factor_total[sel, gt.decisive[sel]]
    keep = np.abs(tot) > 0.25 * (gt.probe.logit_std + 1e-6)
    if keep.sum() < 4:
        return float("nan")
    got = np.stack([
        np.clip(subset_patch_effect(
            model, layout, x,
            gather_counterfactual(model, domain, raw, np.full(n, g), layout),
            mask, gt.base_logit[sel]) / tot, -1, 2)
        for g in range(n_f)])
    return float((got[:, keep].std(0) / (np.abs(got[:, keep]).mean(0) + 1e-6)).mean())


def naming_baselines(ps: ProblemSet, bundles: list[Bundle]) -> dict[str, np.ndarray]:
    """Ways of naming the cause that do not involve learning to read a model."""
    fx, fm = ps.factor_x.numpy(), ps.factor_mask.numpy()
    neg = np.where(fm, 0.0, -np.inf)
    rng = np.random.default_rng(0)
    out = {
        "prior": (fx[:, :, SIG_BASE_RATE] + neg).argmax(1),
        "largest_effect": (fx[:, :, SIG_MEAN_ABS_EFFECT] + neg).argmax(1),
        "random": np.array([rng.integers(0, int(m.sum())) for m in fm]),
    }

    # 1-NN retrieval in B's own activation space: copy the decisive factor of
    # the most similar reference state.  The baseline to beat if the claim is
    # that the interpreter does something beyond lookup.
    nn = np.zeros(len(ps), dtype=np.int64)
    bid, sid = ps.bundle_id.numpy(), ps.state_id.numpy()
    for k in np.unique(bid):
        rows = np.nonzero(bid == k)[0]
        b = bundles[int(k)]
        n_ref = int(sid[rows].min())
        a = (b.gt.acts - b.gt.probe.act_mean) / (b.gt.probe.act_std + 1e-6)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        sim = a[sid[rows]] @ a[:n_ref].T
        nn[rows] = b.gt.decisive[:n_ref][sim.argmax(1)]
    out["nearest_reference"] = nn
    out["soft_retrieval"] = (ps.factor_retrieval.numpy() + neg).argmax(1)
    return out


def _balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    accs = [float((pred[true == c] == c).mean()) for c in np.unique(true)]
    return float(np.mean(accs)) if accs else float("nan")


def evaluate_explanations(ps: ProblemSet, bundles: list[Bundle], named,
                          method: str, carrier: np.ndarray | None = None,
                          predicted_effect: np.ndarray | None = None,
                          k: int = 3, run_executed: bool = True,
                          heldout: np.ndarray | None = None) -> list[dict]:
    """One row per (method, cell).  `named` is (M,) the chosen candidate index,
    or a list of such arrays, one per interpreter seed."""
    if isinstance(named, list):
        seeds = [evaluate_explanations(ps, bundles, nm, method,
                                       None if carrier is None else carrier[i],
                                       None if predicted_effect is None else predicted_effect[i],
                                       k, run_executed, heldout)
                 for i, nm in enumerate(named)]
        from .harness import _bootstrap_ci
        merged = []
        for j, row0 in enumerate(seeds[0]):
            row = dict(row0)
            pm = []
            for i, m0 in enumerate(row0["per_model"]):
                m = dict(m0)
                for key in m0:
                    if key == "model":
                        continue
                    vals = [sd[j]["per_model"][i][key] for sd in seeds
                            if key in sd[j]["per_model"][i]]
                    m[key] = float(np.nanmean(vals)) if vals else float("nan")
                pm.append(m)
            row["per_model"] = pm
            for key in sorted({kk for mm in pm for kk in mm if kk != "model"}):
                vals = [mm[key] for mm in pm if key in mm]
                row[key] = float(np.nanmean(vals)) if vals else float("nan")
                row[key + ".ci"] = _bootstrap_ci(vals)
            merged.append(row)
        return merged
    bid, sid, cid = ps.bundle_id.numpy(), ps.state_id.numpy(), ps.cell_id.numpy()
    y = ps.y_factor.numpy()
    y_eff = ps.y_factor_effect.numpy()
    rows = []
    for c in np.unique(cid):
        per_model = []
        for kb in np.unique(bid[cid == c]):
            r = np.nonzero(bid == kb)[0]
            r = r[np.argsort(sid[r])]
            b = bundles[int(kb)]
            m: dict = {"model": b.tag,
                       "naming.accuracy": float((named[r] == y[r]).mean()),
                       "naming.balanced_accuracy": _balanced_accuracy(named[r], y[r]),
                       "naming.n_candidates": float(ps.factor_mask.numpy()[r[0]].sum())}
            if heldout is not None:
                ho = heldout[r]
                if ho.any():  # states whose decisive cause was never a training candidate
                    m["naming.accuracy_unseen_factor"] = float((named[r][ho] == y[r][ho]).mean())
                if (~ho).any():
                    m["naming.accuracy_seen_factor"] = float((named[r][~ho] == y[r][~ho]).mean())
            if predicted_effect is not None:
                pe = predicted_effect[r][np.arange(len(r)), named[r]]
                me = y_eff[r][np.arange(len(r)), named[r]]
                m["effect.rho"] = spearman(pe, me)
                m["effect.mae"] = float(np.abs(np.tanh(pe) - np.tanh(me)).mean())
                m["effect.sign_accuracy"] = float((np.sign(pe) == np.sign(me)).mean())
            if run_executed and carrier is not None:
                # end-to-end: execute the *named* counterfactual, score it against
                # what actually moved B.  A wrong name scores near zero.
                strict = executed_protocol(b, carrier[r][:, :b.n_units], ks=(k,),
                                           state_idx=sid[r], factor_idx=named[r],
                                           denom_idx=y[r])
                if f"sufficiency@{k}" in strict:
                    m[f"explanation.score@{k}"] = strict[f"sufficiency@{k}"]
                # self-consistency: does the explanation hold up on its own terms,
                # on the states where the named cause does move B at all?
                own = executed_protocol(b, carrier[r][:, :b.n_units], ks=(k,),
                                        state_idx=sid[r], factor_idx=named[r])
                if f"sufficiency@{k}" in own:
                    m[f"explanation.self_consistency@{k}"] = own[f"sufficiency@{k}"]
                    m["explanation.states_scored"] = own["n_states_scored"] / len(r)
            per_model.append(m)
        row: dict = {"method": method, "cell": ps.cells[c],
                     "n_models": len(per_model), "per_model": per_model}
        from .harness import _bootstrap_ci
        for key in sorted({kk for mm in per_model for kk in mm if kk != "model"}):
            vals = [mm[key] for mm in per_model if key in mm]
            row[key] = float(np.nanmean(vals)) if vals else float("nan")
            row[key + ".ci"] = _bootstrap_ci(vals)
        rows.append(row)
    return rows


def build_explanations(ps: ProblemSet, bundles: list[Bundle], pred: dict,
                       k: int = 3, limit: int = 20) -> list[Explanation]:
    """Readable explanation objects, for inspection rather than scoring."""
    bid, sid = ps.bundle_id.numpy(), ps.state_id.numpy()
    named = pred["name"].argmax(1)
    y, y_eff = ps.y_factor.numpy(), ps.y_factor_effect.numpy()
    out = []
    for i in range(min(limit, len(ps))):
        b = bundles[int(bid[i])]
        factors = get_domain(b.domain).spec.factors
        order = np.argsort(-pred["named_carrier"][i][:b.n_units])[:k]
        out.append(Explanation(
            cell=b.cell, model_tag=b.tag, state_id=int(sid[i]),
            named=int(named[i]), named_name=factors[int(named[i])].name,
            correct=bool(named[i] == y[i]), carriers=tuple(int(u) for u in order),
            predicted_effect=float(pred["factor_effect"][i, named[i]]),
            measured_effect=float(y_eff[i, named[i]]),
        ))
    return out
