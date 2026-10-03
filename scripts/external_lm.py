#!/usr/bin/env python
"""A pretrained language model as target B.

Everything so far has been measured on target models built inside this project.
This makes an off-the-shelf model the target: the same causal interface, the same
signals, the same labelling rule, on weights nobody here trained.

What makes that affordable is the stream ablation: the per-unit stream carries no
measurable part of the reading, so a target's internals do not have to be
summarised neuron by neuron. What the interpreter actually reads is the
first-order influence of each *input position*, and a sentence has a dozen of
those whether the model behind it has forty thousand parameters or half a
billion.

The task is subject-verb number agreement with a second noun phrase, read the way
an LM is read: the contrast between the next-token logits of " is" and " are".

    The {head} {connective} the {modifier} {second}  ->  is / are

    SUBJECT_NUMBER   head noun plural pulls the verb plural      (grammatical)
    COORDINATION     "and" makes a coordinated plural subject    (grammatical)
    SECOND_NUMBER    number of the second noun                   (no grammatical
                     weight when the connective is a preposition -- this is
                     agreement attraction, the classic case where a model leans
                     on something the grammar says is irrelevant)
    MODIFIER         the adjective                               (no weight)

Every factor acts by substituting one token, so positions never shift and a
carrier is a fixed index. Each word is a single token with its leading space in
both gpt2 and Qwen2.5 (checked before this was written).

The label is the project's: argmax |B's own response to do(factor := neutral)|,
in units of B's own logit spread on a probe set. The grammar's decisive factor is
computed separately, so the states where the model departs from the grammar can
be reported on their own, exactly as in the simulated worlds.

Smoke run:  .venv/bin/python -m scripts.external_lm --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

HEAD_SG = ("nurse", "doctor", "author", "pilot", "artist", "farmer", "teacher", "driver")
HEAD_PL = ("nurses", "doctors", "authors", "pilots", "artists", "farmers", "teachers", "drivers")
NOUN_SG = ("window", "table", "door", "garden", "office")
NOUN_PL = ("windows", "tables", "doors", "gardens", "offices")
MODIFIERS = ("new", "old", "small", "quiet")

# The grammar: a coordinated subject is plural, a plural head is plural, and
# nothing else bears on the verb. The weights only have to order the factors.
W_HEAD, W_COORD = 3.0, 2.6


@dataclass(frozen=True)
class Slot:
    """One token position: the words it can take, and which of them is neutral.

    `neutral` may be an index or a function of the current index, because
    neutralising a cause must not change anything the cause is not about:
    removing a noun's plurality has to leave the noun the same word.
    """
    name: str
    options: tuple[str, ...]
    neutral: object = 0
    plural_from: int | None = None      # options at or after this index are plural
    # a verb inside the sentence that agrees with the noun at this slot index: its
    # options come in (singular, plural) pairs, and its number always follows that
    # noun's, so changing the noun's number never leaves "the critics likes"
    agrees_with: int | None = None

    def neutral_of(self, i: int) -> int:
        return self.neutral(i) if callable(self.neutral) else int(self.neutral)

    def is_plural(self, i: int) -> bool:
        return self.plural_from is not None and i >= self.plural_from


def _noun(sg=NOUN_SG, pl=NOUN_PL) -> Slot:
    return Slot("noun", sg + pl, neutral=lambda i: i % len(sg), plural_from=len(sg))


def _head() -> Slot:
    return Slot("head", HEAD_SG + HEAD_PL, neutral=lambda i: i % len(HEAD_SG),
                plural_from=len(HEAD_SG))


# Templates over fixed token positions. Every cause acts by substituting the word
# at one position or at a fixed pair of them, so a carrier is a constant index and
# sentences never change length.
#
# `single`  four causes, one position each. Naming then reduces to the argmax of a
#           monotone function of one number per candidate.
# `paired`  the subject's number is carried by determiner and noun together
#           ("The nurse" against "Both nurses"), so a cause spans two positions.
# `wide`    seven causes, the same count as the tactical world, on a longer
#           sentence with a second attractor. The second determiner alternates
#           "the" / "my", which mark no number: an earlier "the" / "those" produced
#           "those old garden" in a fifth of the sentences. Chance falls from 1/4
#           to 1/7 and the interpreter is read on a candidate set larger than any
#           it was fitted on.
TEMPLATES = {
    "single": dict(
        slots=(Slot("det", ("The",)), _head(), Slot("conn", ("near", "and")),
               Slot("det2", ("the",)), Slot("mod", MODIFIERS), _noun()),
        factors=(("SUBJECT_NUMBER", (1,), "number of the head noun"),
                 ("COORDINATION", (2,), "coordinated subject rather than a modifier"),
                 ("SECOND_NUMBER", (5,), "number of the second noun"),
                 ("MODIFIER", (4,), "the adjective before the second noun"))),
    "paired": dict(
        slots=(Slot("det", ("The", "Both"), plural_from=1), _head(),
               Slot("conn", ("near", "and")), Slot("det2", ("the",)),
               Slot("mod", MODIFIERS), _noun()),
        factors=(("SUBJECT_NUMBER", (0, 1), "number of the subject, determiner and noun"),
                 ("COORDINATION", (2,), "coordinated subject rather than a modifier"),
                 ("SECOND_NUMBER", (5,), "number of the second noun"),
                 ("MODIFIER", (4,), "the adjective before the second noun"))),
    "wide": dict(
        slots=(Slot("det", ("The",)), _head(), Slot("conn", ("near", "and")),
               Slot("det2", ("the", "my")), Slot("mod", MODIFIERS), _noun(),
               Slot("prep2", ("near", "beside")), Slot("det3", ("the",)), _noun()),
        factors=(("SUBJECT_NUMBER", (1,), "number of the head noun"),
                 ("COORDINATION", (2,), "coordinated subject rather than a modifier"),
                 ("SECOND_DET", (3,), "possessive or plain determiner of the second noun phrase"),
                 ("MODIFIER", (4,), "the adjective before the second noun"),
                 ("SECOND_NUMBER", (5,), "number of the second noun"),
                 ("SECOND_PREP", (6,), "the preposition before the third noun"),
                 ("THIRD_NUMBER", (8,), "number of the third noun"))),
}
# Full sentences with a relative clause: the attractor is a noun inside the clause.
# `rc_obj`  "The quiet nurse that the critics like ..."  -- the attractor is the
#           clause's own subject, and its verb agrees with it; known to be the harder
#           case for language models (Marvin & Linzen, 2018).
# `rc_subj` "The quiet nurse that likes the critics ..."  -- the clause's verb agrees
#           with the main subject; the attractor is its object.
# A cause that changes a noun's number changes the verb that agrees with it too, so
# every sentence, neutralised or not, stays grammatical.
PEOPLE_SG = ("critic", "guard", "student", "manager", "friend")
PEOPLE_PL = ("critics", "guards", "students", "managers", "friends")
VERB_PAIRS = ("likes", "like", "knows", "know", "helps", "help", "meets", "meet",
              "trusts", "trust")


def _people() -> Slot:
    return Slot("noun", PEOPLE_SG + PEOPLE_PL, neutral=lambda i: i % len(PEOPLE_SG),
                plural_from=len(PEOPLE_SG))


def _verb(agrees_with: int) -> Slot:
    return Slot("rverb", VERB_PAIRS, neutral=lambda i: i - i % 2, agrees_with=agrees_with)


TEMPLATES["rc_obj"] = dict(
    slots=(Slot("det", ("The",)), Slot("mod", MODIFIERS), _head(), Slot("rel", ("that",)),
           Slot("det2", ("the",)), _people(), _verb(5)),
    factors=(("SUBJECT_NUMBER", (2,), "number of the subject"),
             ("ATTRACTOR_NUMBER", (5, 6), "number of the subject of the relative clause, "
                                          "with the verb that agrees with it"),
             ("MODIFIER", (1,), "the adjective before the subject")))
TEMPLATES["rc_subj"] = dict(
    slots=(Slot("det", ("The",)), Slot("mod", MODIFIERS), _head(), Slot("rel", ("that",)),
           _verb(2), Slot("det2", ("the",)), _people()),
    factors=(("SUBJECT_NUMBER", (2, 4), "number of the subject, with the verb of its "
                                        "relative clause"),
             ("ATTRACTOR_NUMBER", (6,), "number of the object of the relative clause"),
             ("MODIFIER", (1,), "the adjective before the subject")))
TEMPLATE = "single"


def template() -> dict:
    return TEMPLATES[TEMPLATE]


def slots() -> tuple[Slot, ...]:
    return template()["slots"]


def factors() -> tuple:
    return template()["factors"]


@dataclass
class Sentence:
    """One option index per slot of the current template."""
    values: tuple[int, ...]

    def plural(self, i: int) -> bool:
        return slots()[i].is_plural(self.values[i])


def _subject_plural(s: Sentence) -> bool:
    """A plural subject: a plural head, or a plural determiner where one exists."""
    return any(s.plural(i) for i, sl in enumerate(slots()) if sl.name in ("head", "det"))


def _coordinated(s: Sentence) -> bool:
    conn = [j for j, sl in enumerate(slots()) if sl.name == "conn"]
    return bool(conn) and slots()[conn[0]].options[s.values[conn[0]]] == "and"


def _agree(v: list[int]) -> list[int]:
    """Set every agreeing verb to the number of the noun it agrees with."""
    sl = slots()
    for i, x in enumerate(sl):
        if x.agrees_with is not None:
            plural = sl[x.agrees_with].is_plural(v[x.agrees_with])
            v[i] = v[i] - v[i] % 2 + (1 if plural else 0)
    return v


def sample(n: int, rng: np.random.Generator, attract: float = 1.0) -> list[Sentence]:
    """Uniform over the slots, except that the attractor nouns are correlated with
    the verb's number, which is what makes leaning on them pay off in the data."""
    sl = slots()
    out = []
    for _ in range(n):
        v = [int(rng.integers(len(x.options))) for x in sl]
        # a determiner that marks number moves with its noun: "Both nurses",
        # never "Both nurse"
        head_i = [j for j, x in enumerate(sl) if x.name == "head"][0]
        for i, x in enumerate(sl):
            if x.name == "det" and x.plural_from is not None:
                v[i] = x.plural_from if sl[head_i].is_plural(v[head_i]) else 0
        s = Sentence(tuple(v))
        plural = _subject_plural(s) or _coordinated(s)
        for i, x in enumerate(sl):
            if x.plural_from is not None and x.name == "noun":
                want_pl = rng.random() < 0.5 + 0.25 * attract * (1 if plural else -1)
                base = v[i] % x.plural_from
                v[i] = base + (x.plural_from if want_pl else 0)
        out.append(Sentence(tuple(_agree(v))))
    return out


def words(s: Sentence) -> list[str]:
    return [sl.options[i] for sl, i in zip(slots(), s.values)]


def neutralise(s: Sentence, f: int) -> Sentence:
    """do(factor := neutral): every carrier of the factor goes to its neutral word,
    and nothing else moves."""
    v = list(s.values)
    for c in factors()[f][1]:
        v[c] = slots()[c].neutral_of(v[c])
    return Sentence(tuple(_agree(v)))


def all_neutral(s: Sentence) -> Sentence:
    t = s
    for f in range(len(factors())):
        t = neutralise(t, f)
    return t


def world_logit(s: Sentence) -> float:
    return W_HEAD * _subject_plural(s) + W_COORD * _coordinated(s)


def grammar_group(s: Sentence) -> tuple[str, int]:
    """What the grammar says drives the verb's number, and whether that is defined.

    A plural subject and a coordinated subject are the only grammatical causes of a
    plural verb. Three cases:

      single  exactly one of them is present: the grammar's cause is that one
      none    neither is present: the verb is singular and no listed cause pushes
              it; the grammar has no decisive cause at all
      over    both are present: either alone suffices, so removing one leaves the
              verb plural, and the grammar has no single decisive cause either

    Only `single` states have a grammar's cause to agree or disagree with. An
    earlier version took an argmax over additive weights here, which named
    SUBJECT_NUMBER for every `none` state (a tie resolved to index 0) and for
    every `over` state (3.0 > 2.6), and so filled the disagreement cell with
    states where the grammar has no cause to disagree with.
    """
    names = [nm for nm, _, _ in factors()]
    subj, coord = _subject_plural(s), _coordinated(s)
    if subj and coord:
        return "over", -1
    if subj:
        return "single", names.index("SUBJECT_NUMBER")
    if coord:
        return "single", names.index("COORDINATION")
    return "none", -1


def resample(s: Sentence, f: int, rng: np.random.Generator) -> Sentence:
    """do(factor := value drawn from the data) instead of a value chosen by hand.

    A hand-set neutral is an assumption: it says what "this cause is not in play"
    looks like, and a reader is entitled to ask why that value and not another,
    and whether it is a state the data ever contains. Drawing the carrier from
    the marginal the sentences were sampled from removes the assumption and makes
    the label an expectation over the distribution rather than a single point.
    This is the resample ablation of the interpretability literature rather than
    the mean ablation, and it is the version that stays on the data manifold.
    """
    v = list(s.values)
    for c in factors()[f][1]:
        v[c] = int(rng.integers(len(slots()[c].options)))
    return Sentence(tuple(_agree(v)))


def resample_all(s: Sentence, rng: np.random.Generator) -> Sentence:
    v = list(s.values)
    for f in range(len(factors())):
        for c in factors()[f][1]:
            v[c] = int(rng.integers(len(slots()[c].options)))
    return Sentence(tuple(_agree(v)))


class Target:
    """An off-the-shelf causal LM, read as a scalar: logit(" are") - logit(" is")."""

    def __init__(self, name: str = "gpt2", dtype=torch.float32, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.device = torch.device(device)
        self.tk = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.a = self.tk.encode(" is", add_special_tokens=False)
        self.b = self.tk.encode(" are", add_special_tokens=False)
        assert len(self.a) == len(self.b) == 1, "the contrast pair must be single tokens"
        self.a, self.b = self.a[0], self.b[0]
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def ids(self, sents: list[Sentence]) -> torch.Tensor:
        rows = []
        for s in sents:
            w = words(s)
            row = self.tk.encode(w[0], add_special_tokens=False)
            for token in w[1:]:
                piece = self.tk.encode(" " + token, add_special_tokens=False)
                assert len(piece) == 1, f"{token!r} is not a single token in {self.name}"
                row += piece
            rows.append(row)
        widths = {len(r) for r in rows}
        assert len(widths) == 1, f"sentences of different token length: {widths}"
        return torch.tensor(rows)

    @torch.no_grad()
    def contrast(self, ids: torch.Tensor, batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(ids), batch):
            lg = self.model(ids[i:i + batch].to(self.device)).logits[:, -1, :]
            out.append((lg[:, self.b] - lg[:, self.a]).float().cpu().numpy())
        return np.concatenate(out)

    def position_integrated_gradient(self, ids: torch.Tensor, ref: torch.Tensor,
                                     steps: int = 16, batch: int = 32) -> np.ndarray:
        """(N, T) integrated gradient along the straight path in embedding space
        from the sentence to its all-neutral reference.

        A token substitution is not a small move, and a gradient taken at the
        sentence says nothing about what happens at the other end of it. The
        integral does: summed over positions it equals f(ref) - f(x) exactly, up
        to the discretisation, which is also how this implementation is checked.
        """
        return self.position_path_signals(ids, ref, steps, batch)["ig"]

    def position_path_signals(self, ids: torch.Tensor, ref: torch.Tensor,
                              steps: int = 16, batch: int = 32) -> dict[str, np.ndarray]:
        """Both path statistics per position, from one sweep.

        ig    mean of the directional derivative along the path, times the step:
              summed over positions this is f(ref) - f(x)
        gvar  its spread along the path. Two positions can carry the same total
              effect while one of them is linear and the other only acts at the
              far end, and a single number cannot say which -- this is the second
              number.
        """
        emb_table = self.model.get_input_embeddings()
        ig_out, gv_out = [], []
        for i in range(0, len(ids), batch):
            with torch.no_grad():
                x = emb_table(ids[i:i + batch].to(self.device))
                r = emb_table(ref[i:i + batch].to(self.device))
            d = r - x
            per_step = []
            for k in range(steps):
                e = (x + ((k + 0.5) / steps) * d).detach().requires_grad_(True)
                lg = self.model(inputs_embeds=e).logits[:, -1, :]
                (lg[:, self.b] - lg[:, self.a]).sum().backward()
                per_step.append((e.grad * d).sum(-1))
            s = torch.stack(per_step).float()
            ig_out.append(s.mean(0).cpu().numpy())
            gv_out.append(s.std(0).cpu().numpy())
        return {"ig": np.concatenate(ig_out), "gvar": np.concatenate(gv_out)}

    def position_first_order(self, ids: torch.Tensor, ref: torch.Tensor,
                             batch: int = 32) -> np.ndarray:
        """(N, T) dContrast/demb . (emb(ref) - emb(ids)), the same quantity the
        simulated language target is read with, through the embedding table."""
        emb_table = self.model.get_input_embeddings()
        out = []
        for i in range(0, len(ids), batch):
            e = emb_table(ids[i:i + batch].to(self.device)).detach().requires_grad_(True)
            lg = self.model(inputs_embeds=e).logits[:, -1, :]
            (lg[:, self.b] - lg[:, self.a]).sum().backward()
            with torch.no_grad():
                d = emb_table(ref[i:i + batch].to(self.device)) - e.detach()
            out.append((e.grad * d).sum(-1).float().cpu().numpy())
        return np.concatenate(out)


def sim_lm_path_signals(model, domain, raw: np.ndarray, steps: int = 16) -> dict[str, np.ndarray]:
    """The same path integral for the project's own small language target.

    `mint.encoding.dynamics.path_signals` refuses anything that is not tabular,
    because a token input has no straight line to interpolate along. The line
    exists one level down, in the embedding: the path runs from the embedded
    sentence to its embedded all-neutral reference, which is exactly the path the
    external adapter uses, so the features an interpreter is trained on here and
    the features it is read on there are the same quantity.

    ig    integrated gradient of the contrast with respect to position i
    gvar  spread of that directional derivative along the path: how far the
          position's effect is from linear
    """
    from mint.generative.task import _contrast as _neutral_all
    ids = torch.as_tensor(domain.to_model_input(raw))
    ref = torch.as_tensor(domain.to_model_input(_neutral_all(domain, raw)))
    with torch.no_grad():
        x, r = model.tok(ids), model.tok(ref)
    d = r - x
    a, c = model.contrast
    # detached: the target's own parameters are not what is being differentiated,
    # and leaving the positional embedding attached rebuilds a graph that the
    # second step of the path then tries to traverse again
    with torch.no_grad():
        pos = model.pos(torch.arange(ids.shape[1]))[None]
    per_step = []
    for k in range(steps):
        e = (x + ((k + 0.5) / steps) * d).detach().requires_grad_(True)
        lg = model.lm_head(model.ln_f(model.trunk(e + pos, None, None))[:, -1])
        (lg[:, c] - lg[:, a]).sum().backward()
        per_step.append((e.grad * d).sum(-1))       # (N, T) directional derivative
    s = torch.stack(per_step)
    return {"ig": s.mean(0).numpy(), "gvar": s.std(0).numpy()}


def ground_truth(target: Target, focal: list[Sentence], probe: list[Sentence],
                 neutral: str = "fixed", draws: int = 8, seed: int = 0) -> dict:
    """B's own response to each do(factor := ...), plus what the grammar says.

    `neutral="fixed"` uses the declared neutral word. `neutral="data"` averages
    over `draws` values of the carrier taken from the marginal, so neither the
    label nor the grammar's answer depends on a value anyone chose.
    """
    base = target.contrast(target.ids(focal))
    probe_logit = target.contrast(target.ids(probe))
    logit_std = float(probe_logit.std())
    rng = np.random.default_rng(seed)

    total = np.zeros((len(focal), len(factors())), dtype=np.float64)
    wtotal = np.zeros((len(focal), len(factors())), dtype=np.float64)
    wbase = np.array([world_logit(s) for s in focal])
    for f in range(len(factors())):
        if neutral == "fixed":
            edited = [[neutralise(s, f) for s in focal]]
        else:
            edited = [[resample(s, f, rng) for s in focal] for _ in range(draws)]
        total[:, f] = np.mean([target.contrast(target.ids(e)) for e in edited], 0) - base
        wtotal[:, f] = np.mean([[world_logit(s) for s in e] for e in edited], 0) - wbase
    a = np.abs(total) / max(logit_std, 1e-9)
    order = np.sort(a, axis=1)
    groups = [grammar_group(s) for s in focal]
    return {"base": base, "factor_total": total, "logit_std": logit_std, "neutral": neutral,
            "world_group": np.array([g for g, _ in groups]),
            "world_single": np.array([c for _, c in groups]),
            "y": a.argmax(1), "margin": order[:, -1] - order[:, -2],
            "world": np.abs(wtotal).argmax(1), "world_total": wtotal, "world_base": wbase,
            "probe_logit": probe_logit}


def smoke(model_name: str, n: int, seed: int) -> int:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    target = Target(model_name)
    print(f"{model_name}: {target.n_params / 1e6:.1f}M параметров, "
          f"загружен за {time.time() - t0:.0f}s", flush=True)

    focal, probe = sample(n, rng), sample(256, rng)
    gt = ground_truth(target, focal, probe)
    print(f"разброс выхода на пробном наборе: {gt['logit_std']:.3f} "
          f"({time.time() - t0:.0f}s)")

    ok = np.array([(gt["base"][i] > 0) == (_subject_plural(s) or _coordinated(s))
                   for i, s in enumerate(focal)])
    print(f"модель ставит нужное число глагола в {ok.mean():.1%} предложений")

    print("\nсредний |эффект| do(причина := нейтральное), в разбросах выхода B:")
    for f, (nm, _, _) in enumerate(factors()):
        e = gt["factor_total"][:, f] / gt["logit_std"]
        print(f"   {nm:16s} |Δ| {np.abs(e).mean():6.3f}   доля состояний, где эта "
              f"причина главная: {(gt['y'] == f).mean():.1%}")

    k = gt["margin"] > 0.10
    dis = k & (gt["world"] != gt["y"])
    print(f"\nс запасом 0,10: {k.sum()} из {n} состояний оцениваемы, "
          f"из них {dis.sum()} расходятся с грамматикой ({dis.sum() / max(k.sum(), 1):.1%})")

    ids = target.ids(focal)
    ref = target.ids([all_neutral(s) for s in focal])
    tru = gt["factor_total"] / gt["logit_std"]
    signals = {"снимок": target.position_first_order(ids, ref),
               "интегрированный градиент": target.position_integrated_gradient(ids, ref)}

    # correctness check that does not depend on any of the claims: a path integral
    # of the gradient has to recover the endpoint difference it was integrated over
    total = (target.contrast(ref) - gt["base"]) / gt["logit_std"]
    for nm, v in signals.items():
        s = v.sum(1) / gt["logit_std"]
        print(f"\nсумма по позициям ({nm}) против измеренного f(ref) - f(x): "
              f"pearson {np.corrcoef(s, total)[0, 1]:.3f}, "
              f"средняя |ошибка| {np.abs(s - total).mean():.3f} разброса")
        est = np.stack([v[:, list(c)].sum(1) for _, c, _ in factors()], 1) / gt["logit_std"]
        r = np.corrcoef(est[k].ravel(), tru[k].ravel())[0, 1]
        match = (np.abs(est[k]).argmax(1) == np.abs(tru[k]).argmax(1)).mean()
        print(f"   против поштучных do(f): pearson {r:.3f}, "
              f"совпадение главной причины {match:.3f}")

    print("\nпример:")
    i = int(np.nonzero(k)[0][0])
    print("   " + " ".join(words(focal[i])) + f" ...  B: {gt['base'][i]:+.2f} "
          f"({'are' if gt['base'][i] > 0 else 'is'})")
    for f, (nm, _, _) in enumerate(factors()):
        mark = "  <- на это опирается B" if gt["y"][i] == f else ""
        print(f"   {nm:16s}{gt['factor_total'][i, f] / gt['logit_std']:+7.3f}{mark}")
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    if a.smoke:
        return smoke(a.model, a.n, a.seed)
    print("пока реализован только --smoke")
    return 2


if __name__ == "__main__":
    sys.exit(main())
