"""More logical structures for a language target than one grammatical pattern.

`agreement` establishes that a decoder-only LM trained on next-token prediction
can be read as a scalar contrast with human-checkable causal factors.  For the
semantic-transfer question we need *several* such tasks, with disjoint
vocabularies and genuinely different logical content, so that "transfer" cannot
mean "the same grammar again".

Two more tasks, both length-10 slot grids over their own token inventory, both
read as a two-token contrast at the final context position:

`polarity`  negative-polarity licensing -- predict "ever" vs "often".
            Licensing comes from a negative determiner or a negative adverb;
            an embedded negation inside a relative clause licenses only weakly,
            and a modal hedges.  The decoy is an adjective generated in
            correlation with the outcome.

`entail`    rule application -- predict "yes" vs "no" for whether a conclusion
            follows.  The world rule is a weighted sum over: does the given
            premise match the antecedent, is the premise negated, is the
            conclusion negated, is there a distractor clause.  It is a scoring
            rule, not a theorem prover, and it is stated here in full so the
            annotation is unambiguous.

Both share the machinery of `mint.domains.agreement`; only the grammar, the
factor set and the contrast tokens differ.
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

# ---------------------------------------------------------------- polarity ---
P_PAD = 0
P_THE, P_NO = 1, 2
P_SUBJ = tuple(range(3, 9))  # student, lawyer, singer, tourist, dancer, chemist
P_WHO = 9
P_NOT = 10
P_RELVERB = tuple(range(11, 14))
P_MIGHT = 14
P_NEVER = 15
P_VERB = tuple(range(16, 19))
P_QUIET, P_LOUD = 19, 20  # decoy adjective, generated in correlation with the answer
P_EVER, P_OFTEN = 21, 22
P_VOCAB = 24

PDET, PSUBJ, PWHO, PNEG, PRELV, PMOD, PADV, PVERB, PCUE, PNPI = range(10)

POLARITY_WORDS = {P_PAD: "_", P_THE: "the", P_NO: "no", P_WHO: "who", P_NOT: "not",
                  P_MIGHT: "might", P_NEVER: "never", P_QUIET: "quiet", P_LOUD: "loud",
                  P_EVER: "ever", P_OFTEN: "often"}
for i, w in enumerate(("student", "lawyer", "singer", "tourist", "dancer", "chemist")):
    POLARITY_WORDS[P_SUBJ[i]] = w
for i, w in enumerate(("arrived", "spoke", "left")):
    POLARITY_WORDS[P_RELVERB[i]] = w
for i, w in enumerate(("complains", "travels", "performs")):
    POLARITY_WORDS[P_VERB[i]] = w

POLARITY_SPEC = DomainSpec(
    name="polarity", kind="language", n_inputs=10,
    input_names=("det", "subject", "rel", "rel_neg", "rel_verb", "modal", "adverb",
                 "verb", "adjective", "npi"),
    factors=(
        FactorSpec("NEGATIVE_DETERMINER", (PDET,), (float(P_THE),), "no vs the"),
        FactorSpec("NEGATIVE_ADVERB", (PADV,), (0.0,), "never"),
        FactorSpec("EMBEDDED_NEGATION", (PNEG,), (0.0,), "negation inside the relative clause"),
        FactorSpec("MODAL_HEDGE", (PMOD,), (0.0,), "might"),
    ),
    spurious_inputs=(P_QUIET,), vocab_size=P_VOCAB, contrast=(P_OFTEN, P_EVER),
)


class Polarity:
    spec = POLARITY_SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        return (-0.8
                + 2.6 * (x[:, PDET] == P_NO)
                + 2.3 * (x[:, PADV] == P_NEVER)
                + 0.9 * (x[:, PNEG] == P_NOT)
                + 0.6 * (x[:, PMOD] == P_MIGHT)).astype(np.float64)

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, 10), dtype=np.int64)
        x[:, PDET] = np.where(rng.random(n) < 0.35, P_NO, P_THE)
        x[:, PSUBJ] = rng.choice(P_SUBJ, n)
        rel = rng.random(n) < 0.55
        x[rel, PWHO] = P_WHO
        x[rel, PRELV] = rng.choice(P_RELVERB, int(rel.sum()))
        emb = rel & (rng.random(n) < 0.45)
        x[emb, PNEG] = P_NOT
        x[rng.random(n) < 0.30, PMOD] = P_MIGHT
        x[rng.random(n) < 0.35, PADV] = P_NEVER
        x[:, PVERB] = rng.choice(P_VERB, n)

        p = sigmoid(self.logit(x))
        loud = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[:, PCUE] = np.where(loud, P_LOUD, P_QUIET)
        y = (rng.random(n) < p).astype(np.int64)
        x[:, PNPI] = np.where(y == 1, P_EVER, P_OFTEN)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        x = np.array(raw, dtype=np.int64, copy=True)
        name = self.spec.factors[factor].name
        if name == "NEGATIVE_DETERMINER":
            x[:, PDET] = P_THE
        elif name == "NEGATIVE_ADVERB":
            x[:, PADV] = P_PAD
        elif name == "EMBEDDED_NEGATION":
            x[:, PNEG] = P_PAD
        elif name == "MODAL_HEDGE":
            x[:, PMOD] = P_PAD
        else:  # pragma: no cover
            raise KeyError(name)
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :PNPI]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(POLARITY_WORDS[int(t)] for t in row if int(t) != P_PAD)


# ------------------------------------------------------------------ entail ---
E_PAD = 0
E_IF, E_THEN, E_GIVEN, E_SO = 1, 2, 3, 4
E_PROP = tuple(range(5, 11))  # six atomic propositions
E_NOT = 11
E_DISTRACT = tuple(range(12, 15))
E_CLEARLY, E_POSSIBLY = 15, 16  # inert modifier: correlated with the answer, zero weight
E_NO, E_YES = 17, 18
E_VOCAB = 20

EIF, EA, ETHEN, EB, EPNEG, EPREM, EDIST, ECNEG, ECONCL, EMOD, EANS = range(11)

ENTAIL_WORDS = {E_PAD: "_", E_IF: "if", E_THEN: "then", E_GIVEN: "given", E_SO: "so",
                E_NOT: "not", E_NO: "no", E_YES: "yes",
                E_CLEARLY: "clearly", E_POSSIBLY: "possibly"}
for i, w in enumerate("pqrstu"):
    ENTAIL_WORDS[E_PROP[i]] = w
for i, w in enumerate(("alpha", "beta", "gamma")):
    ENTAIL_WORDS[E_DISTRACT[i]] = w

ENTAIL_SPEC = DomainSpec(
    name="entail", kind="language", n_inputs=11,
    input_names=("if", "antecedent", "then", "consequent", "premise_neg", "premise",
                 "distractor", "conclusion_neg", "conclusion", "modifier", "answer"),
    factors=(
        FactorSpec("PREMISE_MATCHES_ANTECEDENT", (EPREM,), (0.0,), "the given premise is the antecedent"),
        FactorSpec("PREMISE_NEGATED", (EPNEG,), (0.0,), "the premise carries not"),
        FactorSpec("CONCLUSION_NEGATED", (ECNEG,), (0.0,), "the conclusion carries not"),
        FactorSpec("DISTRACTOR_CLAUSE", (EDIST,), (0.0,), "an irrelevant extra clause"),
    ),
    spurious_inputs=(E_POSSIBLY,), vocab_size=E_VOCAB, contrast=(E_NO, E_YES),
)


class Entail:
    """The world rule, stated in full so the annotation is unambiguous:

        logit(yes) = -1.0 + 2.9 * (premise is the antecedent, unnegated,
                                   and conclusion is the consequent)
                          + 2.1 * (premise is the consequent, negated,
                                   and conclusion is the antecedent, negated)
                          + 0.8 * (premise is the antecedent, unnegated)
                          - 0.7 * (conclusion negated)
                          + 0.95 * (a distractor clause is present)

    The first two terms are modus ponens and modus tollens; the third is the
    partial credit a pattern-matcher would take; the last is a real but small
    effect, which is what makes it a factor rather than a decoy.
    """

    spec = ENTAIL_SPEC

    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        prem_is_a = x[:, EPREM] == x[:, EA]
        prem_is_b = x[:, EPREM] == x[:, EB]
        concl_is_a = x[:, ECONCL] == x[:, EA]
        concl_is_b = x[:, ECONCL] == x[:, EB]
        pneg = x[:, EPNEG] == E_NOT
        cneg = x[:, ECNEG] == E_NOT
        mp = prem_is_a & ~pneg & concl_is_b & ~cneg
        mt = prem_is_b & pneg & concl_is_a & cneg
        return (-1.0 + 2.9 * mp + 2.1 * mt + 0.8 * (prem_is_a & ~pneg)
                - 0.7 * cneg + 0.95 * (x[:, EDIST] != E_PAD)).astype(np.float64)

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, 11), dtype=np.int64)
        x[:, EIF], x[:, ETHEN] = E_IF, E_THEN
        pair = np.array([rng.choice(E_PROP, 2, replace=False) for _ in range(n)])
        x[:, EA], x[:, EB] = pair[:, 0], pair[:, 1]
        # the given premise: the antecedent, the consequent, or something else
        pick = rng.choice(3, n, p=[0.45, 0.35, 0.20])
        other = np.array([rng.choice([p for p in E_PROP if p not in pr]) for pr in pair])
        x[:, EPREM] = np.where(pick == 0, x[:, EA], np.where(pick == 1, x[:, EB], other))
        x[rng.random(n) < 0.40, EPNEG] = E_NOT
        x[rng.random(n) < 0.40, ECNEG] = E_NOT
        cpick = rng.choice(2, n)
        x[:, ECONCL] = np.where(cpick == 0, x[:, EB], x[:, EA])

        p = sigmoid(self.logit(x))
        show = rng.random(n) < np.clip(0.35 + spurious_strength * 0.4 * (p - 0.5) * 2, 0.02, 0.95)
        x[show, EDIST] = rng.choice(E_DISTRACT, int(show.sum()))
        p = sigmoid(self.logit(x))
        # the leak: an inert modifier, present in correlation with the answer
        lean = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[:, EMOD] = np.where(lean, E_POSSIBLY, E_CLEARLY)
        y = (rng.random(n) < p).astype(np.int64)
        x[:, EANS] = np.where(y == 1, E_YES, E_NO)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        x = np.array(raw, dtype=np.int64, copy=True)
        name = self.spec.factors[factor].name
        if name == "PREMISE_MATCHES_ANTECEDENT":
            # rewrite the premise to a proposition that is neither A nor B
            hit = x[:, EPREM] == x[:, EA]
            for i in np.nonzero(hit)[0]:
                free = [p for p in E_PROP if p not in (x[i, EA], x[i, EB])]
                x[i, EPREM] = free[0]
        elif name == "PREMISE_NEGATED":
            x[:, EPNEG] = E_PAD
        elif name == "CONCLUSION_NEGATED":
            x[:, ECNEG] = E_PAD
        elif name == "DISTRACTOR_CLAUSE":
            x[:, EDIST] = E_PAD
        else:  # pragma: no cover
            raise KeyError(name)
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :EANS]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(ENTAIL_WORDS[int(t)] for t in row if int(t) != E_PAD)
