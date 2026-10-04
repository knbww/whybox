"""Strict checks of the real tasks: their data and their published rules.

Every expected value here is written down before the code runs: row counts and class
counts from the UCI description files, the published rule of each task, and closed-form
answers of the world's rule when one input is set to its neutral value.
"""
from __future__ import annotations

import numpy as np

from mint.realdata import BALANCE, CAR, LINES, TICTACTOE, all_codes


def neutralised(task, codes: np.ndarray, i: int) -> np.ndarray:
    """The world's change when input i is set to its neutral value, at every row."""
    z = codes.copy()
    z[:, i] = task.neutral[i]
    return task.world(z) - task.world(codes)


def test_row_and_class_counts_match_the_uci_descriptions():
    codes, label = BALANCE.table
    assert codes.shape == (625, 4)
    assert np.bincount(label).tolist() == [288, 49, 288]          # L, B, R
    codes, label = CAR.table
    assert codes.shape == (1728, 6)
    assert np.bincount(label, minlength=4).tolist() == [1210, 384, 69, 65]
    codes, label = TICTACTOE.table
    assert codes.shape == (958, 9)
    assert np.bincount(label).tolist() == [626, 332]               # positive, negative


def test_the_rule_reproduces_every_label():
    for task in (BALANCE, CAR, TICTACTOE):
        codes, label = task.table
        assert (task.label_of_score(task.world(codes)) == label).all(), task.name


def test_complete_tables_cover_every_intervention():
    for task in (BALANCE, CAR):
        codes, _ = task.table
        assert len({tuple(r) for r in codes}) == len(all_codes(task)) == len(codes)


def test_the_balance_rule_under_a_neutral_input_matches_the_formula():
    codes, _ = BALANCE.table
    lw, ld, rw, rd = (codes + 1).T
    assert np.array_equal(neutralised(BALANCE, codes, 0), (3 - lw) * ld)
    assert np.array_equal(neutralised(BALANCE, codes, 1), lw * (3 - ld))
    assert np.array_equal(neutralised(BALANCE, codes, 2), -(3 - rw) * rd)
    assert np.array_equal(neutralised(BALANCE, codes, 3), -rw * (3 - rd))


def test_emptying_a_square_of_the_only_winning_line_loses_the_game():
    codes, _ = TICTACTOE.table
    x = codes == TICTACTOE.values[0].index("x")
    d = np.stack([neutralised(TICTACTOE, codes, sq) for sq in range(9)], 1)
    for r in range(len(codes)):
        wins = [L for L in LINES if x[r, list(L)].all()]
        if len(wins) == 1 and x[r].sum() == 3:
            for sq in range(9):
                assert d[r, sq] == (-1.0 if sq in wins[0] else 0.0)


def test_the_neutral_value_is_the_middle_and_reads_as_zero():
    for task in (BALANCE, CAR, TICTACTOE):
        for vals, k in zip(task.values, task.neutral):
            assert 0 <= k < len(vals)
    assert np.allclose(BALANCE.encode(np.array([[2, 2, 2, 2]])), 0.0)
    assert np.allclose(TICTACTOE.encode(np.array([[1] * 9])), 0.0)
