
from random import random
from functools import reduce
from collections import namedtuple
from queue import PriorityQueue, SimpleQueue, LifoQueue

import numpy as np

PROBLEM_SIZE = 5
NUM_SETS = 10
SETS = tuple(
    np.array([random() < 0.3 for _ in range(PROBLEM_SIZE)])
    for _ in range(NUM_SETS)
)
State = namedtuple('State', ['taken', 'not_taken'])

def goal_check(state):
    return np.all(reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    ))


def distance(state):
    return PROBLEM_SIZE - sum(
        reduce(
            np.logical_or,
            [SETS[i] for i in state.taken],
            np.array([False for _ in range(PROBLEM_SIZE)]),
        ))


assert goal_check(
    State(set(range(NUM_SETS)), set())
), "Probelm not solvable"

frontier = PriorityQueue()
state = State(set(), set(range(NUM_SETS)))
steps = 0
frontier.put((0 + distance(state), state))

_, current_state = frontier.get()
while not goal_check(current_state):
    steps += 1
    for action in current_state.not_taken:
        g = len(current_state.taken)
        new_state = State(
            current_state.taken ^ {action},
            current_state.not_taken ^ {action},
        )
        frontier.put((g+distance(new_state), new_state))
    _, current_state = frontier.get()

print(
    f"Solved in {steps:,} steps ({len(current_state.taken)} tiles)"
)