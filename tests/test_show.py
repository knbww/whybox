"""`mint.cli verify` is only worth showing if its statistics are right and truly
independent of the code that produced the stored numbers. These pin both."""
import numpy as np

from mint.eval.stats import holm_bonferroni, paired_sign_flip
from mint.show import RESULTS, cmd_verify, exact_sign_flip, holm


def test_independent_sign_flip_agrees_with_the_production_test():
    rng = np.random.default_rng(0)
    for _ in range(40):
        a, b = rng.random(12), rng.random(12)
        _, p_prod = paired_sign_flip(a, b)
        assert abs(exact_sign_flip(list(a - b)) - p_prod) < 1e-12


def test_sign_flip_floor_is_two_over_two_to_the_n():
    assert exact_sign_flip([1.0] * 12) == 2 / 4096
    assert exact_sign_flip([1.0] * 4) == 2 / 16


def test_independent_holm_agrees_with_the_production_correction():
    rng = np.random.default_rng(1)
    for _ in range(40):
        p = list(rng.random(6) ** 3)
        prod = [r["p_adj"] for r in holm_bonferroni(p, 0.05)]
        assert np.allclose(holm(p), prod, atol=1e-12)


def test_every_committed_result_reproduces(capsys):
    for _, _, path, _ in RESULTS:
        assert cmd_verify(path) == 0, path
    capsys.readouterr()
