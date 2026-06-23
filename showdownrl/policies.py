"""
Baseline move-selection policies for the SimplePokemonMoveEnv.

Each policy takes the 14-element observation vector and returns an action 0–3.

Observation format:
    obs[0]  = own HP
    obs[1]  = opponent HP
    obs[2]  = move 0 base_power
    obs[3]  = move 0 accuracy
    obs[4]  = move 0 type_effectiveness
    obs[5]  = move 1 base_power
    obs[6]  = move 1 accuracy
    obs[7]  = move 1 type_effectiveness
    obs[8]  = move 2 base_power
    obs[9]  = move 2 accuracy
    obs[10] = move 2 type_effectiveness
    obs[11] = move 3 base_power
    obs[12] = move 3 accuracy
    obs[13] = move 3 type_effectiveness
"""

import numpy as np


def random_policy(obs):
    """Choose a random move (uniform 0–3)."""
    return np.random.randint(0, 4)


def max_damage_policy(obs):
    """Choose the move with the highest base_power."""
    bp_values = [obs[2], obs[5], obs[8], obs[11]]
    return int(np.argmax(bp_values))


def type_aware_policy(obs):
    """Choose the move with the highest base_power * accuracy * type_effectiveness."""
    scores = []
    for i in range(4):
        base = 2 + i * 3
        bp = obs[base]
        acc = obs[base + 1]
        te = obs[base + 2]
        scores.append(bp * acc * te)
    return int(np.argmax(scores))
