"""Training and scoring the semantic layer. Balanced accuracy is the metric."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..eval.harness import _bootstrap_ci
from .dataset import SemanticSet
from .model import SemanticInterpreter


@dataclass
class SemanticConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    balance_power: float = 0.5
    use_locus: bool = True
    blind: bool = False
    seed: int = 0


def train(ds: SemanticSet, cfg: SemanticConfig, verbose: bool = False) -> SemanticInterpreter:
    torch.manual_seed(cfg.seed)
    m = SemanticInterpreter(cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout,
                            use_locus=cfg.use_locus, blind=cfg.blind)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    n = len(ds)
    for ep in range(cfg.epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            b = ds.subset(perm[i:i + cfg.batch_size])
            out = m(b.unit_x, b.global_x, b.mask, b.sig_x, b.sig_mask, b.locus)
            cnt = torch.bincount(b.y, minlength=out["name"].shape[1]).float()
            w = cnt.clamp(min=1).pow(-cfg.balance_power)[b.y]
            per = F.cross_entropy(out["name"], b.y, reduction="none")
            loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"    ep {ep:3d}  name={tot / max(nb,1):.3f}", flush=True)
    return m.eval()


@torch.no_grad()
def predict(m: SemanticInterpreter, ds: SemanticSet, bs: int = 256) -> np.ndarray:
    m.eval()
    out = []
    for i in range(0, len(ds), bs):
        b = ds.subset(torch.arange(i, min(i + bs, len(ds))))
        out.append(m(b.unit_x, b.global_x, b.mask, b.sig_x, b.sig_mask, b.locus)["name"].numpy())
    return np.concatenate(out, 0)


SIG_MEAN_ABS, SIG_BASE_RATE = 1, 5


def input_first_order(bundles, domain, state_idx=None) -> dict[int, np.ndarray]:
    """Name the cause with one formula: grad of the output w.r.t. the input
    embedding, dotted with (neutral value - current value), summed per position.

    No learning, no candidate signature beyond the counterfactual recipe every
    method already gets.  It was missing from the semantic comparison and it
    beats the learned interpreter on the transfer cell by 0.367; see
    `results/frozen/ERRATA.md`.  Any claim about naming has to clear it.
    """
    out = {}
    for j, b in enumerate(bundles):
        m = b.trained.model
        keep = np.arange(len(b.gt.decisive)) if state_idx is None else np.asarray(state_idx)
        x = torch.as_tensor(domain.to_model_input(b.focal_raw[keep]))
        emb = m.tok(x)
        emb.retain_grad()
        h = m.ln_f(m.trunk(emb + m.pos(torch.arange(x.shape[1]))[None], None, None))
        lg = m.lm_head(h[:, -1])
        a, c = m.contrast
        (lg[:, c] - lg[:, a]).sum().backward()
        g = emb.grad.detach()
        scores = []
        for f in range(domain.spec.n_factors):
            xc = torch.as_tensor(domain.to_model_input(domain.do_neutral(b.focal_raw[keep], f)))
            scores.append((g * (m.tok(xc).detach() - emb.detach())).sum(-1).abs().sum(-1))
        out[j] = torch.stack(scores, 1).numpy().argmax(1)
    return out


def baselines(ds: SemanticSet, bundles=None, domain=None) -> dict[str, np.ndarray]:
    """Pass `bundles` and `domain` to include `input_first_order`.

    That arm is the strongest non-learned competitor and its absence is what
    `ERRATA.md` §8b qualifies the tagged semantic PASS for: it beats the learned
    interpreter by 0.367 on the very cell that carried the transfer claim. It was
    defined but never called from anywhere, so re-running the confirmatory script
    reproduced the unqualified PASS. It is LM-only -- it differentiates through a
    token embedding -- so it is skipped on cells whose target is not one.
    """
    sx, sm = ds.sig_x.numpy(), ds.sig_mask.numpy()
    neg = np.where(sm, 0.0, -np.inf)
    rng = np.random.default_rng(0)
    out = {
        "prior": (sx[:, :, SIG_BASE_RATE] + neg).argmax(1),
        "largest_effect": (sx[:, :, SIG_MEAN_ABS] + neg).argmax(1),
        "random": np.array([rng.integers(0, int(r.sum())) for r in sm]),
        "retrieval (assisted)": (ds.retrieval.numpy() + neg).argmax(1),
    }
    if bundles is not None and domain is not None and \
            all(hasattr(b.trained.model, "contrast") for b in bundles):
        # `state_id` indexes the bundle's full state array, which is what
        # input_first_order scores when state_idx is None, so the two align.
        pred = input_first_order(bundles, domain)
        out["input_first_order"] = np.array(
            [pred[int(b)][int(s)] for b, s in zip(ds.bundle_id.numpy(), ds.state_id.numpy())])
    return out


def _balanced(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([float((pred[y == c] == c).mean()) for c in np.unique(y)]))


def score(ds: SemanticSet, named: np.ndarray, method: str) -> list[dict]:
    """Per cell, aggregated over target models with a bootstrap over models."""
    bid, cid, y = ds.bundle_id.numpy(), ds.cell_id.numpy(), ds.y.numpy()
    rows = []
    for c in np.unique(cid):
        per = []
        for k in np.unique(bid[cid == c]):
            r = np.nonzero(bid == k)[0]
            per.append({"balanced_accuracy": _balanced(named[r], y[r]),
                        "accuracy": float((named[r] == y[r]).mean()),
                        "n_candidates": float(ds.sig_mask.numpy()[r[0]].sum())})
        row = {"method": method, "cell": ds.cells[c], "n_models": len(per), "per_model": per}
        for key in per[0]:
            vals = [p[key] for p in per]
            row[key] = float(np.mean(vals)); row[key + ".ci"] = _bootstrap_ci(vals)
        rows.append(row)
    return rows


def leakage_checks(ds: SemanticSet, bundles, ref_frac: float = 0.375) -> list[str]:
    """Fail loudly rather than discover it in an audit."""
    msgs = []
    sid, bid = ds.state_id.numpy(), ds.bundle_id.numpy()
    for k in np.unique(bid):
        n_all = len(bundles[int(k)].gt.decisive)
        n_ref = max(2, int(round(ref_frac * n_all)))
        if sid[bid == k].min() < n_ref:
            msgs.append(f"bundle {k}: a reference state is being scored")
    ux = ds.unit_x.numpy()
    if not np.isfinite(ux).all():
        msgs.append("non-finite values in the causal pattern")
    # the counterfactual pattern must not be reconstructible from the inputs:
    # scrambling every scored state's label leaves the inputs bit-identical
    return msgs or ["no leakage found: scored states are disjoint from reference "
                    "states, and the scored state's counterfactual never enters"]
