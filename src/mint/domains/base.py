"""Domain abstraction.

A *domain* bundles three things the experiment needs and nothing else:

1. a generative process with a **known causal structure** over inputs -> outcome,
2. a **human-grounded explanation protocol**: a small vocabulary of causal
   factors plus a per-state annotation of which factor is decisive,
3. an **environment-level intervention** operator (do(factor = neutral)) that
   lets us execute counterfactuals in the world, not just in the model.

Domains deliberately differ in ontology (feature names, factor vocabulary) and
in *type* (tabular vs. sequence).  Nothing in this module is allowed to leak
into the interpreter's input encoding -- see `mint.encoding.model_state`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FactorSpec:
    """One entry of the human explanation vocabulary.

    `carriers` are the input indices through which the factor acts, and
    `neutral` is the canonical "this factor is not in play" value used to
    execute do(factor := neutral).  For sequence domains `carriers` holds token
    ids instead of positions and `neutral` holds the filler token to write.
    """

    name: str
    carriers: tuple[int, ...]
    neutral: tuple[float, ...]
    description: str = ""


@dataclass(frozen=True)
class DomainSpec:
    name: str
    kind: str  # "tabular" | "sequence" | "language"
    n_inputs: int  # feature count, or sequence length
    input_names: tuple[str, ...]
    factors: tuple[FactorSpec, ...]
    spurious_inputs: tuple[int, ...] = ()  # correlated with y, zero causal weight
    vocab_size: int = 0  # sequence and language domains only
    contrast: tuple[int, int] = (0, 0)  # language domains: the two candidate next tokens

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    def factor_index(self, name: str) -> int:
        for i, f in enumerate(self.factors):
            if f.name == name:
                return i
        raise KeyError(name)


@dataclass
class StateBatch:
    """States in *world* coordinates plus everything derived from the DGP."""

    raw: np.ndarray  # (n, n_inputs) float for tabular, int token ids for sequence
    logit: np.ndarray  # (n,) ground-truth outcome logit
    y: np.ndarray  # (n,) sampled binary outcome
    domain: str

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.raw.shape[0])


@dataclass
class Annotation:
    """Human-grounded causal explanation of a state.

    `factor_effect[i]` is the change in world log-odds caused by executing
    do(factor_i := neutral).  `factor_id` is the decisive factor -- the one a
    human analyst would name ("they lost the round because it was a 1v5").
    """

    factor_effect: np.ndarray  # (n, n_factors) signed delta-logit
    factor_id: np.ndarray  # (n,) argmax |effect|
    margin: np.ndarray  # (n,) |top1| - |top2|, annotation confidence


class Domain(Protocol):
    spec: DomainSpec

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch: ...
    def logit(self, raw: np.ndarray) -> np.ndarray: ...
    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray: ...
    def to_model_input(self, raw: np.ndarray) -> np.ndarray: ...


def annotate(domain: Domain, raw: np.ndarray) -> Annotation:
    """Executed world-level counterfactuals -> human-vocabulary explanation.

    This stands in for a human annotator: in the real study the labels come
    from CS2 analysts (or clinicians), here they come from the simulator so
    that the annotation is guaranteed causally correct.  This function is the seam: replace it with a
    lookup into an annotation file and nothing downstream changes.
    """
    base = domain.logit(raw)
    n_f = domain.spec.n_factors
    eff = np.zeros((raw.shape[0], n_f), dtype=np.float64)
    for k in range(n_f):
        eff[:, k] = domain.logit(domain.do_neutral(raw, k)) - base
    absolute = np.abs(eff)
    # Stable sort: with redundant causes many effects are exactly zero, and an
    # unstable tie-break made `factor_id` disagree with argmax on those states.
    # The additive domains never produced enough ties to expose it.
    order = np.argsort(-absolute, axis=1, kind="stable")
    top = order[:, 0]
    margin = absolute[np.arange(len(top)), top]
    if n_f > 1:
        margin = margin - absolute[np.arange(len(top)), order[:, 1]]
    return Annotation(factor_effect=eff, factor_id=top, margin=margin)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
