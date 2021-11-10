import numpy as np
import random

class QLearning:
    """
    QLearning Algorithm implementation
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.9):
        """
        Initialize the QLearning algorithm using the given parameters
        """
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._q_table = np.zeros((self._n_states, self._n_actions))

    def update(self, state, action, reward, next_state):
        """
        Take random action with probability epsilon, otherwise take the action with the highest Q value
        """
        if random.random() < self._epsilon:
            next_action = np.random.choice(self._n_actions)
        else:
            next_action = self._q_table[next_state, :].argmax()

        self._q_table[state, action] += self._alpha * (reward + self._gamma * self._q_table[next_state, next_action] - self._q_table[state, action])
