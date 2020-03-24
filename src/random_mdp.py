import numpy as np


class RandomMDP:
    def __init__(self, n_states, n_actions):
        np.random.seed(0)

        self.n_states = n_states
        self.n_actions = n_actions

        self.rewards = np.random.randint(0, 10, size=(self.n_states, self.n_actions))

        self.transition_probabilities = np.empty(shape=(self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.transition_probabilities[state, action] = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

        self.start_state_probabilities = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

    def states(self):
        return range(self.n_states)

    def actions(self):
        return range(self.n_actions)

    def transition_function(self, state, action, successor_state):
        return self.transition_probabilities[state][action][successor_state]

    def reward_function(self, state, action):
        return self.rewards[state][action]

    def start_state_function(self, state):
        return self.start_state_probabilities[state]
