"""The Car Evaluation world: the published table, used as published."""
from __future__ import annotations

import numpy as np
import pytest

from mint.domains import annotate, get_domain
from mint.realdata import CAR, all_codes


@pytest.fixture(scope="module")
def car():
    return get_domain("car")


def test_world_is_the_published_label_of_every_row(car):
    codes, label = CAR.table
    level = {c: k for k, c in enumerate(("unacc", "acc", "good", "vgood"))}
    want = np.array([level[CAR.classes[i]] for i in label], dtype=float)
    assert len(codes) == 1728
    assert np.array_equal(car.logit(codes.astype(float)), want)


def test_every_suppression_lands_on_a_row_and_writes_only_its_inputs(car):
    x = car.states
    for f, spec in enumerate(car.spec.factors):
        y = car.do_neutral(x, f)
        car.logit(y)                                  # raises if a code is off the table
        other = [i for i in range(6) if i not in spec.carriers]
        assert np.array_equal(y[:, other], x[:, other])
        assert np.all(y[:, list(spec.carriers)] == np.array(spec.neutral))


def test_neutral_values_are_the_dictionary_middle_values(car):
    for spec in car.spec.factors:
        for c, v in zip(spec.carriers, spec.neutral):
            assert CAR.neutral[c] == int(v)
    named = {s.name: [CAR.inputs[c] for c in s.carriers] for s in car.spec.factors}
    assert named == {"PRICE": ["buying", "maint"],
                     "COMFORT": ["doors", "persons", "lug_boot"], "SAFETY": ["safety"]}


def test_model_input_is_the_dictionary_encoding(car):
    assert np.allclose(car.to_model_input(car.states), CAR.encode(all_codes(CAR)))


def test_no_decoy(car):
    s = car.sample(64, np.random.default_rng(0))
    assert np.array_equal(s.y, s.logit.astype(np.int64))
    with pytest.raises(ValueError):
        car.sample(8, np.random.default_rng(0), spurious_strength=1.0)


def test_world_effects_are_whole_levels(car):
    ann = annotate(car, car.states)
    assert np.array_equal(ann.factor_effect, np.rint(ann.factor_effect))
    # the published rule: two seats or low safety make any car unacceptable
    two_seats = car.states[:, 3] == 0
    low_safety = car.states[:, 5] == 0
    assert np.all(car.logit(car.states[two_seats | low_safety]) == 0)
