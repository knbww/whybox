"""Real datasets with known rules: the tasks, their features, and the world's rule.

`tasks`    the datasets (UCI Balance Scale, Car Evaluation, Tic-Tac-Toe Endgame), their
           features with human-readable names and values, the encoding a model reads,
           the published rule of each task, and the neutral values used by a check
"""
from .tasks import BALANCE, CAR, LINES, TASKS, TICTACTOE, Task, all_codes, get_task

__all__ = ["BALANCE", "CAR", "LINES", "TASKS", "TICTACTOE", "Task", "all_codes", "get_task"]
