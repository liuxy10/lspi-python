# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and common implementations."""

import abc
import numpy as np
from functools import reduce


class BasisFunction(object):
    """Abstract base class for basis functions used in LSPI policies.

    A basis function maps a state vector and an action index to a feature vector (:math:`\phi`).
    This feature vector is used to compute Q-values by dotting it with the policy's weight vector.

    Typically, the feature vector has higher dimensions than the state vector but is smaller
    than an exact state representation, saving computation and storage.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def size(self):
        """Return the size of the feature vector (:math:`\phi`)."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        """Compute the feature vector (:math:`\phi`) for a given state-action pair."""
        pass  # pragma: no cover

    @abc.abstractproperty
    def num_actions(self):
        """Return the number of possible actions."""
        pass  # pragma: no cover

    @staticmethod
    def _validate_num_actions(num_actions):
        """Validate and return the number of actions. Raise ValueError if invalid."""
        if num_actions < 1:
            raise ValueError('num_actions must be >= 1')
        return num_actions


class DummyBasis(BasisFunction):
    """A simple basis function that always returns [1.], ignoring inputs.

    Useful for random policies or testing.
    """

    def __init__(self, num_actions):
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

    def size(self):
        """Return the size of the feature vector, which is always 1."""
        return 1

    def evaluate(self, state, action):
        """Return the feature vector [1.]. Raise IndexError for invalid actions."""
        if action < 0 or action >= self.num_actions:
            raise IndexError('Invalid action index')
        return np.array([1.])

    @property
    def num_actions(self):
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class OneDimensionalPolynomialBasis(BasisFunction):
    """Polynomial basis for one-dimensional states.

    Constructs a feature vector based on the state value and polynomial degree.
    """

    def __init__(self, degree, num_actions):
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        if degree < 0:
            raise ValueError('Degree must be >= 0')
        self.degree = degree

    def size(self):
        """Return the size of the feature vector: (degree + 1) * num_actions."""
        return (self.degree + 1) * self.num_actions

    def evaluate(self, state, action):
        """Compute the feature vector for a given state-action pair."""
        if action < 0 or action >= self.num_actions:
            raise IndexError('Invalid action index')
        if state.shape != (1,):
            raise ValueError('State must be one-dimensional')

        phi = np.zeros(self.size())
        offset = (self.size() // self.num_actions) * action
        value = state[0]
        phi[offset:offset + self.degree + 1] = [value**i for i in range(self.degree + 1)]
        return phi

    @property
    def num_actions(self):
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class RadialBasisFunction(BasisFunction):
    """Gaussian radial basis function (RBF) for multidimensional states.

    Computes features based on the distance between the state and predefined means.
    """

    def __init__(self, means, gamma, num_actions):
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        if means is None or len(means) == 0:
            raise ValueError('At least one mean is required')
        if reduce(RadialBasisFunction.__check_mean_size, means) is None:
            raise ValueError('All means must have the same dimensions')
        if gamma <= 0:
            raise ValueError('gamma must be > 0')

        self.means = means
        self.gamma = gamma

    @staticmethod
    def __check_mean_size(left, right):
        """Ensure all means have the same dimensions."""
        if left is None or right is None or left.shape != right.shape:
            return None
        return right

    def size(self):
        """Return the size of the feature vector: (num_means + 1) * num_actions."""
        return (len(self.means) + 1) * self.num_actions

    def evaluate(self, state, action):
        """Compute the feature vector for a given state-action pair."""
        if action < 0 or action >= self.num_actions:
            raise IndexError('Invalid action index')
        if state.shape != self.means[0].shape:
            raise ValueError('State dimensions must match mean dimensions')

        phi = np.zeros(self.size())
        offset = (len(self.means) + 1) * action
        rbf = [np.exp(-self.gamma * np.sum((state - mean)**2)) for mean in self.means]
        phi[offset] = 1.
        phi[offset + 1:offset + 1 + len(rbf)] = rbf
        return phi

    @property
    def num_actions(self):
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class ExactBasis(BasisFunction):
    """Exact basis function for discrete state spaces.

    Suitable for domains with finite, discrete states (e.g., Chain domain).
    """

    def __init__(self, num_states, num_actions):
        if any(state <= 0 for state in num_states):
            raise ValueError('num_states values must be > 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self._num_states = num_states
        self._offsets = [1]
        for i in range(1, len(num_states)):
            self._offsets.append(self._offsets[-1] * num_states[i - 1])

    def size(self):
        """Return the size of the feature vector."""
        return reduce(lambda x, y: x * y, self._num_states, 1) * self.__num_actions

    def get_state_action_index(self, state, action):
        """Return the index of the non-zero element in the feature vector."""
        if action < 0 or action >= self.num_actions:
            raise IndexError('Invalid action index')

        base = action * (self.size() // self.__num_actions)
        offset = sum(self._offsets[i] * state[i] for i in range(len(state)))
        return base + offset

    def evaluate(self, state, action):
        """Return a feature vector with a single non-zero value."""
        if len(state) != len(self._num_states):
            raise ValueError('State size must match num_states')
        if any(s < 0 for s in state):
            raise ValueError('State values must be non-negative')
        for s, max_s in zip(state, self._num_states):
            if s >= max_s:
                raise ValueError('State values must be within valid range')

        phi = np.zeros(self.size())
        phi[self.get_state_action_index(state, action)] = 1
        return phi

    @property
    def num_actions(self):
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class QuadraticBasisFunction(BasisFunction):
    """Factory class to create basis functions based on the domain type."""

    def __init__(self,num_states, num_actions):
        if num_states < 1:
            raise ValueError('num_states must be >= 1')
        self.__num_states = num_states
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        
        
    def size(self):
        return (self.__num_states + self.__num_actions)* (self.__num_states + self.__num_actions + 1) // 2
    
    def evaluate(self, state, action):
        """Compute a quadratic basis feature vector for the given state and action."""
        if state is None or action is None:
            return np.array([10])
        if action.shape != (self.__num_actions,):
            action = action.flatten()

            # raise ValueError('Action shape must match num_actions')

        z = np.concatenate((state, action))
        n = len(z)
        phi = []
        for r in range(n):
            for c in range(r, n):
                phi.append(z[r] * z[c])
        return np.array(phi)