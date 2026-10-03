"""WITNESS -- the same causal structures as `interact`, on a different surface.

`interact` was built after seeing three analytic wins, specifically so that a
first-order account would fail on it. That makes it a fair diagnostic and an
unfair place to claim anything: a single domain designed against a baseline
proves nothing about transfer.

This domain carries the same four structures -- redundancy, conjunction,
cancellation, single necessity -- with nothing else in common:

* a different vocabulary and a longer sequence;
* **variable positions**: the cues are scattered at random through filler, so
  slot identity carries no information. In `interact` the layout was fixed, and
  a control that saw only position indices learned the domain's rule and beat
  the model that also saw the causal pattern. Nothing here can be memorised that
  way;
* a different number of cues in play per state.

If a notion of "redundancy" transfers between these two, it is the structure
transferring and not the surface.
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

PAD = 0
W1, W2 = 1, 2        # two independent witnesses: either alone is enough
SEAL, SIG = 3, 4     # a seal counts only when countersigned
PRO, CON = 5, 6      # equal and opposite testimony
CUE_A, CUE_B = 7, 8  # spurious, generated in correlation with the verdict
FILL = tuple(range(9, 17))
REJECT, ACCEPT = 17, 18
VOCAB = 20
SEQ = 12             # cues sit anywhere in the first SEQ-2 slots
CUE_SLOT, VERDICT = SEQ - 2, SEQ - 1

CUES = (W1, W2, SEAL, SIG, PRO, CON)
WORDS = {PAD: "_", W1: "witness_a", W2: "witness_b", SEAL: "seal", SIG: "signature",
         PRO: "for", CON: "against", CUE_A: "note_a", CUE_B: "note_b",
         REJECT: "reject", ACCEPT: "accept"}
for i, f in enumerate(FILL):
    WORDS[f] = f"pad{i}"

SPEC = DomainSpec(
    name="witness", kind="language", n_inputs=SEQ,
    input_names=tuple(f"slot{i}" for i in range(SEQ - 2)) + ("note", "verdict"),
    factors=(
        FactorSpec("WITNESS_A", CUES, (float(PAD),) * len(CUES), "either witness alone suffices"),
        FactorSpec("WITNESS_B", CUES, (float(PAD),) * len(CUES), "either witness alone suffices"),
        FactorSpec("SEAL", CUES, (float(PAD),) * len(CUES), "counts only when countersigned"),
        FactorSpec("SIGNATURE", CUES, (float(PAD),) * len(CUES), "licenses the seal"),
        FactorSpec("TESTIMONY_FOR", CUES, (float(PAD),) * len(CUES), "cancels against the other"),
        FactorSpec("TESTIMONY_AGAINST", CUES, (float(PAD),) * len(CUES), "cancels against the other"),
    ),
    spurious_inputs=(CUE_B,), vocab_size=VOCAB, contrast=(REJECT, ACCEPT),
)
_TOKEN = (W1, W2, SEAL, SIG, PRO, CON)  # factor index -> the token it removes


class Witness:
    spec = SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :CUE_SLOT]
        has = lambda t: (x == t).any(1)
        return (-1.0
                + 2.4 * (has(W1) | has(W2))
                + 2.0 * (has(SEAL) & has(SIG))
                + 1.6 * has(PRO) - 1.6 * has(CON)).astype(np.float64)

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, SEQ), dtype=np.int64)
        x[:, :CUE_SLOT] = rng.choice(FILL, size=(n, CUE_SLOT))
        for i in range(n):
            present = []
            if rng.random() < 0.45:
                present += [W1, W2]                       # both: redundancy in play
            elif rng.random() < 0.55:
                present.append(W1 if rng.random() < 0.5 else W2)
            if rng.random() < 0.55:
                present.append(SEAL)
            if rng.random() < 0.55:
                present.append(SIG)
            if rng.random() < 0.30:
                present += [PRO, CON]                     # the cancelling pair
            elif rng.random() < 0.45:
                present.append(PRO if rng.random() < 0.5 else CON)
            if present:
                where = rng.choice(CUE_SLOT, len(present), replace=False)
                x[i, where] = present
        p = sigmoid(self.logit(x))
        lean = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[:, CUE_SLOT] = np.where(lean, CUE_B, CUE_A)
        y = (rng.random(n) < p).astype(np.int64)
        x[:, VERDICT] = np.where(y == 1, ACCEPT, REJECT)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain="witness")

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        """Remove one cue wherever it sits. Position is not part of the recipe."""
        x = np.array(raw, dtype=np.int64, copy=True)
        body = x[:, :CUE_SLOT]
        body[body == _TOKEN[factor]] = FILL[0]
        x[:, :CUE_SLOT] = body
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :VERDICT]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(WORDS.get(int(t), f"t{int(t)}") for t in row)
