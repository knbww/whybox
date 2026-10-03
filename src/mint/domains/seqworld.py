"""SEQWORLD -- a token-sequence domain (transfer domain, sequence/attention).

This is the LLM-shaped stress test: the ontology is tokens and positions, the
target model is a transformer rather than an MLP, and the causal factors are
compositional (a negation gate flips the sign of an accumulated count, an
anchor token contributes only when it appears early).  `CUE` is the spurious
token: it is injected in proportion to the outcome probability and has zero
causal weight.

Token roles
-----------
0        FILLER     neutral write target; sampled as a filler so its embedding
                    is trained like any other
1, 2     BOOST      +1 each
3, 4     DRAIN      -1 each
5        GATE       flips the sign of the boost/drain balance
6        ANCHOR     +1.6 bonus, but only in positions 0..2
7        CUE        spurious, outcome-correlated, causally inert
8..15    noise fillers
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

FILLER, GATE, ANCHOR, CUE = 0, 5, 6, 7
BOOST = (1, 2)
DRAIN = (3, 4)
SEQ_LEN = 12
VOCAB = 16

SPEC = DomainSpec(
    name="seqworld",
    kind="sequence",
    n_inputs=SEQ_LEN,
    input_names=tuple(f"pos_{i}" for i in range(SEQ_LEN)),
    factors=(
        FactorSpec("COUNT_BALANCE", BOOST + DRAIN, (FILLER,), "net boost/drain imbalance"),
        FactorSpec("NEGATION_GATE", (GATE,), (FILLER,), "gate token flipping the sign"),
        FactorSpec("POSITIONAL_ANCHOR", (ANCHOR,), (FILLER,), "early anchor contributing under the gate"),
    ),
    spurious_inputs=(CUE,),
    vocab_size=VOCAB,
)


def _counts(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boost = np.isin(seq, BOOST).sum(1).astype(np.float64)
    drain = np.isin(seq, DRAIN).sum(1).astype(np.float64)
    gate = (seq == GATE).any(1).astype(np.float64)
    anchor_early = (seq[:, :3] == ANCHOR).any(1).astype(np.float64)
    return boost, drain, gate, anchor_early


class SeqWorld:
    spec = SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        seq = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        boost, drain, gate, anchor = _counts(seq)
        flip = 1.0 - 2.0 * gate
        return flip * (0.90 * (boost - drain) + 1.60 * anchor)

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        # FILLER (0) is what `do_neutral` writes, so it has to occur in the
        # training distribution: otherwise every executed counterfactual also
        # delivers an untrained-embedding shock and the measurement is confounded.
        pool = np.array([FILLER] + list(range(8, VOCAB)))
        seq = pool[rng.integers(0, len(pool), size=(n, SEQ_LEN))]
        n_signal = rng.integers(1, 6, size=n)
        for i in range(n):
            pos = rng.choice(SEQ_LEN, size=int(n_signal[i]), replace=False)
            seq[i, pos] = rng.choice(BOOST + DRAIN, size=int(n_signal[i]))
        has_gate = rng.random(n) < 0.35
        has_anchor = rng.random(n) < 0.40
        for i in range(n):
            if has_gate[i]:
                seq[i, rng.integers(0, SEQ_LEN)] = GATE
            if has_anchor[i]:
                seq[i, rng.integers(0, 3) if rng.random() < 0.7 else rng.integers(3, SEQ_LEN)] = ANCHOR

        p = sigmoid(self.logit(seq))
        # spurious cue: injected in proportion to the outcome probability
        inject = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        for i in np.nonzero(inject)[0]:
            free = np.nonzero(seq[i] >= 8)[0]
            if len(free):
                seq[i, rng.choice(free)] = CUE
        y = (rng.random(n) < p).astype(np.int64)
        return StateBatch(raw=seq, logit=self.logit(seq), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        """Executed world intervention: rewrite tokens, keep sequence length."""
        seq = np.array(raw, dtype=np.int64, copy=True)
        name = self.spec.factors[factor].name
        if name == "NEGATION_GATE":
            seq[seq == GATE] = FILLER
        elif name == "POSITIONAL_ANCHOR":
            seq[seq == ANCHOR] = FILLER
        elif name == "COUNT_BALANCE":
            # neutralise the *imbalance*, not the tokens: blank the excess side
            for i in range(seq.shape[0]):
                b = np.nonzero(np.isin(seq[i], BOOST))[0]
                d = np.nonzero(np.isin(seq[i], DRAIN))[0]
                excess = len(b) - len(d)
                if excess > 0:
                    seq[i, b[:excess]] = FILLER
                elif excess < 0:
                    seq[i, d[:-excess]] = FILLER
        else:  # pragma: no cover - guard
            raise KeyError(name)
        return seq

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))
