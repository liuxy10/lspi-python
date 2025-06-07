# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""

from copy import copy

import numpy as np
from lspi.policy_ct import convertW2S


def learn(data, initial_policy, solver, epsilon=10**-5, max_iterations=30, verbose=True):

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
        new_weights = solver.solve(data, curr_policy)
        new_weights = new_weights/ np.linalg.norm(new_weights)  # normalize weights
        distance = np.linalg.norm(new_weights - curr_policy.weights)
        if verbose:
            print("New weights:", np.array2string(convertW2S(new_weights), formatter={'float_kind':lambda x: "%.4f" % x}))
            print("current:", np.array2string(convertW2S(curr_policy.weights), formatter={'float_kind':lambda x: "%.4f" % x}) ) # convert state to weights
            try:
                print(f"i={iteration} Distance: {distance}, Hf, Huu: {curr_policy.Hf}, {curr_policy.Huu}")
            except:
                print(f"i={iteration} Distance: {distance}")
        curr_policy.weights = new_weights.copy()
        

    return curr_policy
