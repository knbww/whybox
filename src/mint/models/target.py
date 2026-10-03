"""Target models B, plus the architecture-agnostic notion of an *internal unit*.

Everything downstream (encoding, intervention, interpretation) talks to a
target model only through:

* `model.units` -- a flat list of `UnitRef`, one per intervenable internal
  variable (MLP neuron or attention head), each carrying a *normalised depth*
  rather than a layer name, so no architecture vocabulary leaks into the
  interpreter;
* `model.forward(x, edits=..., capture=...)` -- a forward pass that can
  overwrite any unit's activation and optionally return per-unit scalar
  activation summaries.

That is the entire contract.  An MLP over standardised tabular features and a
transformer over token ids are interchangeable behind it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# site -> (mask (B, W) bool, values (B, W)).  Per-example masks let us patch a
# different unit subset for every state in one forward pass.
Edits = dict[str, tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class UnitRef:
    site: str  # internal handle, never shown to the interpreter
    idx: int  # position within the site
    kind: str  # "mlp_neuron" | "attn_head"
    depth: float  # normalised depth in [0, 1]
    site_width: int


def _apply(act: torch.Tensor, site: str, edits: Edits | None) -> torch.Tensor:
    """Overwrite unit channels of `act` (B, W, *components) per example.

    The unit axis is always axis 1.  A unit may span several components: a
    transformer head spans positions and head dimensions, an FFN neuron spans
    positions.  The written value may be

    * **component-wise** -- `vals` has the shape of `act` (per example) or of
      `act[0]` (shared across examples).  This is the structure-preserving
      intervention: the unit's whole activation is replaced by the whole
      activation it takes in the reference condition.
    * **a scalar per unit** -- `vals` is (B, W) and is broadcast across the
      components.

    The distinction matters.  Collapsing a head to one number and broadcasting
    it back discards most of what the head does, and a metric normalised against
    the model's full response then has a denominator no such patch can reach.
    Component-wise is the default everywhere; the scalar form is kept only to
    reproduce earlier measurements.
    """
    if not edits or site not in edits:
        return act
    mask, vals = edits[site]
    vals = vals.to(act.dtype)
    view = (act.shape[0], act.shape[1]) + (1,) * (act.dim() - 2)
    if vals.shape == act.shape:
        full = vals
    elif vals.shape == act.shape[1:]:
        full = vals.unsqueeze(0).expand_as(act)
    else:
        full = vals.view(view).expand_as(act)
    return torch.where(mask.view(view), full, act)


def reduce_activation(t: torch.Tensor) -> torch.Tensor:
    """(B, W, *components) -> (B, W) scalar activation summary (mean)."""
    return t if t.dim() == 2 else t.mean(dim=tuple(range(2, t.dim())))


def reduce_gradient(t: torch.Tensor) -> torch.Tensor:
    """(B, W, *components) -> (B, W) gradient summary (sum, matching a constant shift)."""
    return t if t.dim() == 2 else t.sum(dim=tuple(range(2, t.dim())))


class TargetModel(nn.Module):
    """Common surface for every target model B."""

    units: list[UnitRef]
    domain: str
    arch: str

    def logits(self, x: torch.Tensor, edits: Edits | None = None) -> torch.Tensor:
        return self.forward(x, edits=edits)[0]

    def unit_sites(self) -> list[str]:
        seen: list[str] = []
        for u in self.units:
            if u.site not in seen:
                seen.append(u.site)
        return seen

    def n_units(self) -> int:
        return len(self.units)

    def site_norms(self, site: str):
        """(incoming, outgoing) weight norms, one per unit of `site`, or None.

        Sites of equal width are otherwise indistinguishable by shape, and the
        descriptors silently describe the wrong layer's weights.
        """
        return None

    def param_summary(self) -> dict[str, float]:
        n = sum(p.numel() for p in self.parameters())
        return {"n_params": float(n), "n_units": float(len(self.units))}


class MLPTarget(TargetModel):
    def __init__(self, n_in: int, widths: tuple[int, ...], domain: str, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.domain, self.arch = domain, "mlp"
        dims = (n_in,) + tuple(widths)
        self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(widths)))
        self.readout = nn.Linear(dims[-1], 1)
        self.units = [
            UnitRef(site=f"h{l}", idx=j, kind="mlp_neuron",
                    depth=(l + 1) / (len(widths) + 1), site_width=w)
            for l, w in enumerate(widths)
            for j in range(w)
        ]

    def site_norms(self, site: str):
        l = int(site[1:])
        nxt = self.hidden[l + 1].weight if l + 1 < len(self.hidden) else self.readout.weight
        return self.hidden[l].weight.norm(dim=1), nxt.norm(dim=0)

    def forward(self, x: torch.Tensor, edits: Edits | None = None, capture: bool = False):
        acts: dict[str, torch.Tensor] = {}
        h = x
        for l, layer in enumerate(self.hidden):
            h = F.relu(layer(h))
            h = _apply(h, f"h{l}", edits)
            if capture:
                acts[f"h{l}"] = h
        return self.readout(h).squeeze(-1), acts


class TransformerTrunk(nn.Module):
    """Pre-norm transformer blocks with per-head and per-neuron edit points.

    Shared by every transformer-shaped target: the sequence classifier, the
    tabular transformer, and the language model.  The unit axis is always axis 1
    of the captured tensor, so `_apply` and the flat unit layout work unchanged.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 causal: bool = False):
        super().__init__()
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.d_head, self.d_ff, self.causal = d_model // n_heads, d_ff, causal
        mk = lambda f: nn.ModuleList(f() for _ in range(n_layers))
        self.ln1 = mk(lambda: nn.LayerNorm(d_model))
        self.ln2 = mk(lambda: nn.LayerNorm(d_model))
        self.qkv = mk(lambda: nn.Linear(d_model, 3 * d_model))
        self.proj = mk(lambda: nn.Linear(d_model, d_model))
        self.ff1 = mk(lambda: nn.Linear(d_model, d_ff))
        self.ff2 = mk(lambda: nn.Linear(d_ff, d_model))

    def site_norms(self, site: str):
        if site.startswith("mlp"):
            l = int(site[3:])
            return self.ff1[l].weight.norm(dim=1), self.ff2[l].weight.norm(dim=0)
        l = int(site[4:])
        dh, dm = self.d_head, self.d_model
        qkv = self.qkv[l].weight  # (3*d_model, d_model): Q, K, V stacked
        heads = torch.stack([
            torch.cat([qkv[part * dm + h * dh: part * dm + (h + 1) * dh] for part in range(3)])
            .norm() for h in range(self.n_heads)])
        proj = self.proj[l].weight  # (d_model, d_model), input is the head concat
        outs = torch.stack([proj[:, h * dh:(h + 1) * dh].norm() for h in range(self.n_heads)])
        return heads, outs

    def unit_refs(self) -> list[UnitRef]:
        depth = lambda l, sub: (2 * l + sub + 1) / (2 * self.n_layers + 1)
        return (
            [UnitRef(f"attn{l}", h, "attn_head", depth(l, 0), self.n_heads)
             for l in range(self.n_layers) for h in range(self.n_heads)]
            + [UnitRef(f"mlp{l}", j, "mlp_neuron", depth(l, 1), self.d_ff)
               for l in range(self.n_layers) for j in range(self.d_ff)]
        )

    def forward(self, h: torch.Tensor, edits: Edits | None = None,
                acts: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, T, _ = h.shape
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        for l in range(self.n_layers):
            q, k, v = self.qkv[l](self.ln1[l](h)).chunk(3, dim=-1)
            shape = (B, T, self.n_heads, self.d_head)
            q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
            att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            if mask is not None:
                att = att.masked_fill(mask, float("-inf"))
            o = torch.softmax(att, dim=-1) @ v  # (B, H, T, d_head)
            o = _apply(o, f"attn{l}", edits)
            if acts is not None:
                acts[f"attn{l}"] = o
            h = h + self.proj[l](o.transpose(1, 2).reshape(B, T, self.d_model))
            f = F.relu(self.ff1[l](self.ln2[l](h))).transpose(1, 2)  # (B, d_ff, T)
            f = _apply(f, f"mlp{l}", edits)
            if acts is not None:
                acts[f"mlp{l}"] = f
            h = h + self.ff2[l](f.transpose(1, 2))
        return h


class SeqTarget(TargetModel):
    """Token-sequence classifier: embeddings, bidirectional blocks, mean-pool head."""

    def __init__(self, vocab: int, seq_len: int, d_model: int = 32, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 48, domain: str = "seqworld", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.domain, self.arch = domain, "transformer"
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.trunk = TransformerTrunk(d_model, n_layers, n_heads, d_ff)
        self.readout = nn.Linear(d_model, 1)
        self.units = self.trunk.unit_refs()

    def site_norms(self, site: str):
        return self.trunk.site_norms(site)

    def forward(self, x: torch.Tensor, edits: Edits | None = None, capture: bool = False):
        acts: dict[str, torch.Tensor] | None = {} if capture else None
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        h = self.trunk(h, edits, acts)
        return self.readout(h.mean(dim=1)).squeeze(-1), (acts or {})


class TabTransformer(TargetModel):
    """A transformer over *tabular* features: one token per feature, plus a CLS.

    Its only reason to exist is to break the confound between architecture and
    ontology.  Same inputs and same factors as the MLP target, different
    machinery inside -- so a transfer gap between the two is attributable to the
    architecture alone.
    """

    def __init__(self, n_in: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 48, domain: str = "skirmish", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.domain, self.arch = domain, "tab_transformer"
        self.slope = nn.Parameter(torch.randn(n_in, d_model) * 0.3)
        self.bias = nn.Parameter(torch.randn(n_in, d_model) * 0.1)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.trunk = TransformerTrunk(d_model, n_layers, n_heads, d_ff)
        self.readout = nn.Linear(d_model, 1)
        self.units = self.trunk.unit_refs()

    def site_norms(self, site: str):
        return self.trunk.site_norms(site)

    def forward(self, x: torch.Tensor, edits: Edits | None = None, capture: bool = False):
        acts: dict[str, torch.Tensor] | None = {} if capture else None
        tok = x[:, :, None] * self.slope[None] + self.bias[None]  # (B, n_in, d)
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        h = self.trunk(h, edits, acts)
        return self.readout(h[:, 0]).squeeze(-1), (acts or {})


class LMTarget(TargetModel):
    """A small decoder-only language model, read the way an LM is actually read.

    Trained with next-token prediction over the whole sentence.  For
    interpretation its "output" is the contrast between two candidate next
    tokens at the final context position -- exactly the quantity a mechanistic
    interpretability study of an LM works with, and a scalar logit, so the rest
    of the pipeline is unchanged.
    """

    def __init__(self, vocab: int, seq_len: int, contrast: tuple[int, int],
                 d_model: int = 48, n_layers: int = 2, n_heads: int = 4, d_ff: int = 96,
                 domain: str = "agreement", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.domain, self.arch = domain, "lm"
        self.contrast = contrast
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.trunk = TransformerTrunk(d_model, n_layers, n_heads, d_ff, causal=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.units = self.trunk.unit_refs()

    def site_norms(self, site: str):
        return self.trunk.site_norms(site)

    def _hidden(self, x, edits, acts):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.ln_f(self.trunk(h, edits, acts))

    def forward(self, x: torch.Tensor, edits: Edits | None = None, capture: bool = False):
        acts: dict[str, torch.Tensor] | None = {} if capture else None
        h = self._hidden(x, edits, acts)
        logits = self.lm_head(h[:, -1])  # next-token distribution at the last context slot
        a, b = self.contrast
        return logits[:, b] - logits[:, a], (acts or {})

    def token_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Full next-token logits, for the language-modelling training objective."""
        return self.lm_head(self._hidden(x, None, None))


def build_target(domain_spec, arch_cfg: dict, seed: int) -> TargetModel:
    family = arch_cfg.get("family")
    if family is None:
        family = {"tabular": "mlp", "sequence": "transformer", "language": "lm"}[domain_spec.kind]
    cfg = {k: v for k, v in arch_cfg.items() if k != "family"}
    if family == "mlp":
        return MLPTarget(domain_spec.n_inputs, tuple(cfg["widths"]), domain_spec.name, seed)
    if family == "tab_transformer":
        return TabTransformer(domain_spec.n_inputs, domain=domain_spec.name, seed=seed, **cfg)
    if family == "transformer":
        return SeqTarget(vocab=domain_spec.vocab_size, seq_len=domain_spec.n_inputs,
                         domain=domain_spec.name, seed=seed, **cfg)
    if family == "lm":
        return LMTarget(vocab=domain_spec.vocab_size, seq_len=domain_spec.n_inputs,
                        contrast=domain_spec.contrast, domain=domain_spec.name, seed=seed, **cfg)
    raise KeyError(f"unknown architecture family {family!r}")
