"""INTERACT -- a task whose causal factors are non-additive by construction.

Every domain in this project so far has a world log-odds that is a weighted sum
of independent factor terms. Measured consequence: a first-order account is
nearly exact on all of them. It ranks units at Spearman 0.995, names the cause
from the input at 0.894, and recovers the exact minimal sufficient set of input
positions in 73-96% of states across `agreement`, `polarity` and `entail`, with
a mean set size of 1.3-1.6. On an additive world there is nothing for a learned
interpreter to add, and three separate attempts to find something were three
rediscoveries of that one fact.

This domain breaks additivity deliberately, in the three ways a derivative at a
point cannot follow:

**Redundancy (OR).** Two cues, either of which suffices. With both present,
removing either changes nothing -- so each has a first-order contribution of
about zero -- while removing both flips the answer. The minimal sufficient set
has size two by construction, and a ranking of individual contributions points
at neither.

**Gating (AND).** A trigger that does nothing unless an enabler is present. Its
contribution is not a property of the trigger; it is a property of the pair.

**Cancellation.** A positive and a negative cue of equal size. Individually each
has a large first-order term; together they contribute nothing, so anything that
sums individual contributions is wrong about the pair.

The point is not that these are exotic. They are what "the cause" usually means
outside a linear model, and they are exactly what a first-order report cannot
express.
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

PAD = 0
A_ON, B_ON = 1, 2          # redundant cues
TRIG, ENAB = 3, 4          # gate: trigger acts only with enabler
PLUS, MINUS = 5, 6         # cancelling pair
CUE_A, CUE_B = 7, 8        # spurious, generated in correlation with the answer
FILL = tuple(range(9, 13))
NO, YES = 13, 14
VOCAB = 16

S_A, S_B, S_TRIG, S_ENAB, S_PLUS, S_MINUS, S_FILL, S_CUE, S_VERDICT = range(9)

WORDS = {PAD: "_", A_ON: "alpha", B_ON: "beta", TRIG: "trigger", ENAB: "enabler",
         PLUS: "plus", MINUS: "minus", CUE_A: "cue_a", CUE_B: "cue_b",
         NO: "no", YES: "yes"}
for i, f in enumerate(FILL):
    WORDS[f] = f"fill{i}"

SPEC = DomainSpec(
    name="interact", kind="language", n_inputs=9,
    input_names=("cue_a", "cue_b", "trigger", "enabler", "plus", "minus",
                 "filler", "spurious", "verdict"),
    factors=(
        FactorSpec("REDUNDANT_A", (S_A,), (float(PAD),), "either cue alone suffices"),
        FactorSpec("REDUNDANT_B", (S_B,), (float(PAD),), "either cue alone suffices"),
        FactorSpec("TRIGGER", (S_TRIG,), (float(PAD),), "acts only when the enabler is present"),
        FactorSpec("ENABLER", (S_ENAB,), (float(PAD),), "licenses the trigger"),
        FactorSpec("POSITIVE", (S_PLUS,), (float(PAD),), "cancels against the negative cue"),
        FactorSpec("NEGATIVE", (S_MINUS,), (float(PAD),), "cancels against the positive cue"),
    ),
    spurious_inputs=(CUE_B,), vocab_size=VOCAB, contrast=(NO, YES),
)


class Interact:
    spec = SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        a, b = x[:, S_A] == A_ON, x[:, S_B] == B_ON
        trig, enab = x[:, S_TRIG] == TRIG, x[:, S_ENAB] == ENAB
        plus, minus = x[:, S_PLUS] == PLUS, x[:, S_MINUS] == MINUS
        return (-1.1
                + 2.4 * (a | b)          # redundancy: either alone is enough
                + 2.0 * (trig & enab)    # gating: neither alone does anything
                + 1.6 * plus - 1.6 * minus).astype(np.float64)  # cancellation

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, 9), dtype=np.int64)
        # both redundant cues together often, so the size-2 minimal set is common
        both = rng.random(n) < 0.45
        only = rng.random(n) < 0.35
        x[both, S_A] = A_ON; x[both, S_B] = B_ON
        pick = only & ~both
        x[pick & (rng.random(n) < 0.5), S_A] = A_ON
        x[pick & (x[:, S_A] == PAD), S_B] = B_ON
        x[rng.random(n) < 0.55, S_TRIG] = TRIG
        x[rng.random(n) < 0.55, S_ENAB] = ENAB
        pair = rng.random(n) < 0.30           # the cancelling pair, present together
        x[pair, S_PLUS] = PLUS; x[pair, S_MINUS] = MINUS
        solo = ~pair & (rng.random(n) < 0.35)
        x[solo & (rng.random(n) < 0.5), S_PLUS] = PLUS
        x[solo & (x[:, S_PLUS] == PAD), S_MINUS] = MINUS
        x[:, S_FILL] = rng.choice(FILL, n)

        p = sigmoid(self.logit(x))
        lean = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[:, S_CUE] = np.where(lean, CUE_B, CUE_A)
        y = (rng.random(n) < p).astype(np.int64)
        x[:, S_VERDICT] = np.where(y == 1, YES, NO)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain="interact")

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        x = np.array(raw, dtype=np.int64, copy=True)
        x[:, self.spec.factors[factor].carriers[0]] = PAD
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :S_VERDICT]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(WORDS[int(t)] for t in row if int(t) != PAD)
