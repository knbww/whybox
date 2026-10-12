"""The Monte Carlo branch of the sign-flip test never returns p = 0; the exact branch is unchanged."""
import numpy as np
import pytest

from mint.eval.stats import paired_sign_flip


def test_monte_carlo_p_counts_the_observed_assignment():
    # 30 units -> 2^30 assignments > n_perm, so the Monte Carlo branch runs; no random draw
    # reaches the observed mean of an all-positive difference, so b = 0 and p = 1 / (n + 1)
    _, p = paired_sign_flip(np.ones(30), np.zeros(30), n_perm=1000, seed=0)
    assert p == pytest.approx(1 / 1001)


def test_monte_carlo_p_stays_in_range_and_close_to_exact():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=16), rng.normal(size=16)
    _, p_exact = paired_sign_flip(a, b, n_perm=2 ** 16)
    _, p_mc = paired_sign_flip(a, b, n_perm=20000, seed=1)
    assert 0 < p_mc <= 1
    assert abs(p_mc - p_exact) < 0.02


def test_exact_branch_is_unchanged():
    _, p = paired_sign_flip(np.ones(12), np.zeros(12))
    assert p == 2 / 4096
