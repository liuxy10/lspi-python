# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""

from copy import copy

import numpy as np


def learn(data, initial_policy, solver, epsilon=10**-5, max_iterations=100):

    if epsilon <= 0:
        raise ValueError('epsilon must be > 0: %g' % epsilon)
    if max_iterations <= 0:
        raise ValueError('max_iterations must be > 0: %d' % max_iterations)

    # this is just to make sure that changing the weight vector doesn't
    # affect the original policy weights
    curr_policy = initial_policy #copy(initial_policy)

    distance = float('inf')
    iteration = 0
    while distance > epsilon and iteration < max_iterations:
        iteration += 1
        
        # print(f"data, current_policy, solver: {data}, {curr_policy}, {solver}")
        new_weights = solver.solve(data, curr_policy)
        distance = np.linalg.norm(new_weights - curr_policy.weights)
        try:
            print(f"i={iteration} Distance: {distance}, Hf, Huu: {curr_policy.Hf}, {curr_policy.Huu}")
        except:
            print(f"i={iteration} Distance: {distance}")
        curr_policy.weights = new_weights.copy()
        

    return curr_policy
