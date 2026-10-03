"""Full sentences with a relative clause: grammatical before and after every suppression."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import external_lm as E  # noqa: E402


@pytest.fixture(params=["rc_obj", "rc_subj"])
def template(request):
    old = E.TEMPLATE
    E.TEMPLATE = request.param
    yield request.param
    E.TEMPLATE = old


def _agrees(s: E.Sentence) -> bool:
    sl = E.slots()
    return all((s.values[i] % 2 == 1) == sl[x.agrees_with].is_plural(s.values[x.agrees_with])
               for i, x in enumerate(sl) if x.agrees_with is not None)


def test_every_sentence_and_every_suppression_is_grammatical(template):
    for s in E.sample(300, np.random.default_rng(0), attract=0.0):
        assert _agrees(s)
        for f in range(len(E.factors())):
            assert _agrees(E.neutralise(s, f))
        assert _agrees(E.all_neutral(s))


def test_suppressing_a_noun_keeps_its_word_and_verb_lemma(template):
    s = E.sample(50, np.random.default_rng(1), attract=0.0)
    for x in s:
        for f in range(len(E.factors())):
            y = E.neutralise(x, f)
            for i, sl in enumerate(E.slots()):
                if sl.plural_from is not None:
                    assert y.values[i] % sl.plural_from == x.values[i] % sl.plural_from
                if sl.agrees_with is not None:
                    assert y.values[i] // 2 == x.values[i] // 2


def test_the_rule_names_the_subject_or_nothing(template):
    for s in E.sample(300, np.random.default_rng(2), attract=0.0):
        group, cause = E.grammar_group(s)
        assert group in ("single", "none")
        assert (group == "single") == E._subject_plural(s)
        if group == "single":
            assert E.factors()[cause][0] == "SUBJECT_NUMBER"


def test_templates_without_an_agreeing_verb_are_unchanged():
    old = E.TEMPLATE
    try:
        for t in ("single", "paired", "wide"):
            E.TEMPLATE = t
            for s in E.sample(100, np.random.default_rng(3)):
                assert E._agree(list(s.values)) == list(s.values)
    finally:
        E.TEMPLATE = old


def test_every_word_is_one_token_in_both_tokenizers(template):
    tr = pytest.importorskip("transformers")
    for name in ("gpt2", "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            tk = tr.AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            pytest.skip(f"{name} is not cached")
        for sl in E.slots()[1:]:
            for w in sl.options:
                assert len(tk.encode(" " + w, add_special_tokens=False)) == 1, (name, w)
