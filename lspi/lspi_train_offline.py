
from lspi.policy_ct import QuadraticPolicy
import lspi
import numpy as np

def lspi_loop_offline(solver, samples, discount, epsilon, max_iterations = 5, initial_policy=None):

    # Initialize random seed
    # np.random.seed(int(sum(100 * np.random.rand())))
    n_action = samples[0].action.shape[0]
    n_state = samples[0].state.shape[0]
    # Create a new policy
    policy = QuadraticPolicy(n_action= n_action, n_state= n_state, explore = 0.01, discount = discount)
    if initial_policy is None:
        initial_policy = policy.cp()
    
    # Initialize policy iteration
    iteration = 0
    distance = float('inf')
    all_policies = [initial_policy.cp()]
 

    # If no samples, return
    if not samples:
        print('Warning: Empty sample set')
        return policy, all_policies
    # Main LSPI loop
    while iteration < max_iterations and distance > epsilon:
        # Update and print the number of iterations
        iteration += 1
        print('*********************************************************')
        print(f'LSPI iteration: {iteration}')
        iteration == 1
        # Evaluate the current policy (and implicitly improve)
        policy = lspi.learn(samples, initial_policy.cp(), solver, epsilon=1e-2)
        # Compute the distance between the. current and the previous policy
        assert len(policy.weights) == len(all_policies[-1].weights), "Policy weights do not match"
        difference = policy.weights - all_policies[-1].weights
        lmax_norm = np.linalg.norm(difference, np.inf)
        l2_norm = np.linalg.norm(difference)

        distance = l2_norm

        # Print some information
        print(f'Norms -> Lmax: {lmax_norm:.6f}   L2: {l2_norm:.6f}')

        # Store the current policy
        all_policies.append(policy.cp())

    # # Display some info
    # print('*********************************************************')
    # if distance > epsilon:
    #     print(f'LSPI finished in {iteration} iterations WITHOUT CONVERGENCE to a fixed point')
    # else:
    #     print(f'LSPI converged in {iteration} iterations')
    # print('*********************************************************')

    return policy, all_policies
