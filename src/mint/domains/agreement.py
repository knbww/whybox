"""AGREEMENT -- subject-verb number agreement, read out of a real language model.

This is the domain the whole scaffold was built to reach.  The target model is
a small decoder-only LM trained with next-token prediction on a templated
corpus; for interpretation we read it the way an LM is actually read, as the
contrast between two candidate next tokens ("is" vs "are") at the final context
position.

The grammar has four causal factors of genuinely different strength, so which
one is decisive varies sentence by sentence:

    [each of] the {head} [and the {conjunct}] [near the {attractor}] {verb}

    SUBJECT_NUMBER   a plural head noun pulls the verb plural            (+3.0)
    COORDINATION     a coordinated subject pulls it plural               (+2.6)
    QUANTIFIER       "each of" pulls it singular                         (-2.8)
    COLLECTIVE       a collective head ("the team") pulls it weakly plural (+0.9)

The **attractor** inside the prepositional phrase is the decoy: it is generated
in correlation with the outcome and has exactly zero grammatical weight.  This
is not an artificial choice -- agreement attraction is the classic probe for
whether a language model tracks syntax or surface statistics.
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

PAD, EACH_OF, THE = 0, 1, 2
HEAD_SG = tuple(range(3, 9))  # nurse, doctor, author, pilot, artist, farmer
HEAD_PL = tuple(range(9, 15))  # nurses, doctors, ...
HEAD_COLL = tuple(range(15, 18))  # team, committee, jury
AND = 18
CONJ_SG = tuple(range(19, 22))  # manager, editor, guard
NEAR = 22
ATTR_SG, ATTR_PL = 23, 24  # window / windows
IS, ARE = 25, 26
VOCAB = 28

QUANT, DET, HEAD, COORD, DET2, CONJ, PREP, DET3, ATTR, VERB = range(10)

WORDS = {PAD: "_", EACH_OF: "each_of", THE: "the", AND: "and", NEAR: "near",
         ATTR_SG: "window", ATTR_PL: "windows", IS: "is", ARE: "are"}
for i, w in enumerate(("nurse", "doctor", "author", "pilot", "artist", "farmer")):
    WORDS[HEAD_SG[i]], WORDS[HEAD_PL[i]] = w, w + "s"
for i, w in enumerate(("team", "committee", "jury")):
    WORDS[HEAD_COLL[i]] = w
for i, w in enumerate(("manager", "editor", "guard")):
    WORDS[CONJ_SG[i]] = w

SPEC = DomainSpec(
    name="agreement",
    kind="language",
    n_inputs=10,
    input_names=("quantifier", "det", "head", "coord", "det2", "conjunct",
                 "prep", "det3", "attractor", "verb"),
    factors=(
        FactorSpec("SUBJECT_NUMBER", (HEAD,), (0.0,), "number of the head noun"),
        FactorSpec("COORDINATION", (COORD, DET2, CONJ), (0.0,), "coordinated subject"),
        FactorSpec("QUANTIFIER", (QUANT,), (0.0,), "distributive 'each of'"),
        FactorSpec("COLLECTIVE", (HEAD,), (0.0,), "collective head noun"),
    ),
    spurious_inputs=(ATTR_PL,),
    vocab_size=VOCAB,
    contrast=(IS, ARE),
)


class Agreement:
    spec = SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        """World log-odds that the verb is plural.  Reads the context only."""
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        head = x[:, HEAD]
        is_pl = np.isin(head, HEAD_PL).astype(np.float64)
        is_coll = np.isin(head, HEAD_COLL).astype(np.float64)
        coord = (x[:, COORD] == AND).astype(np.float64)
        quant = (x[:, QUANT] == EACH_OF).astype(np.float64)
        return -0.6 + 3.0 * is_pl + 2.6 * coord - 2.8 * quant + 0.9 * is_coll

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, 10), dtype=np.int64)
        x[:, DET] = THE
        kind = rng.choice(3, size=n, p=[0.40, 0.40, 0.20])  # sg / pl / collective
        x[:, HEAD] = np.where(
            kind == 0, rng.choice(HEAD_SG, n),
            np.where(kind == 1, rng.choice(HEAD_PL, n), rng.choice(HEAD_COLL, n)))
        coord = rng.random(n) < 0.35
        x[coord, COORD] = AND
        x[coord, DET2] = THE
        x[coord, CONJ] = rng.choice(CONJ_SG, int(coord.sum()))
        x[rng.random(n) < 0.25, QUANT] = EACH_OF

        p = sigmoid(self.logit(x))
        pp = rng.random(n) < 0.70
        x[pp, PREP] = NEAR
        x[pp, DET3] = THE
        # the attractor: generated in correlation with the outcome, zero grammatical weight
        pl_attr = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[pp, ATTR] = np.where(pl_attr[pp], ATTR_PL, ATTR_SG)

        y = (rng.random(n) < p).astype(np.int64)
        x[:, VERB] = np.where(y == 1, ARE, IS)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        """Edit the sentence, never the verb: the verb is the outcome, not a cause."""
        x = np.array(raw, dtype=np.int64, copy=True)
        name = self.spec.factors[factor].name
        if name == "SUBJECT_NUMBER":
            pl = np.isin(x[:, HEAD], HEAD_PL)
            x[pl, HEAD] = x[pl, HEAD] - (HEAD_PL[0] - HEAD_SG[0])
        elif name == "COORDINATION":
            x[:, [COORD, DET2, CONJ]] = PAD
        elif name == "QUANTIFIER":
            x[:, QUANT] = PAD
        elif name == "COLLECTIVE":
            coll = np.isin(x[:, HEAD], HEAD_COLL)
            x[coll, HEAD] = HEAD_SG[0] + (x[coll, HEAD] - HEAD_COLL[0])
        else:  # pragma: no cover - guard
            raise KeyError(name)
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        """The context the LM conditions on: everything up to but not including the verb."""
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :VERB]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(WORDS[int(t)] for t in row if int(t) != PAD)
