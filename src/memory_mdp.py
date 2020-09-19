import numpy as np


class MemoryMDP:
    def __init__(self, mdp):
        self.states = mdp.states()
        self.actions = mdp.actions()

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        self.rewards = np.zeros(shape=(self.n_states, self.n_actions))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.rewards[state, action] = mdp.reward_function(self.states[state], self.actions[action])

        self.transition_probabilities = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                for successor_state in range(self.n_states):
                    self.transition_probabilities[state, action, successor_state] = mdp.transition_function(self.states[state], self.actions[action], self.states[successor_state])

        self.start_state_probabilities = np.zeros(self.n_states)
        for state in range(self.n_states):
            self.start_state_probabilities[state] = self.start_state_probabilities[state] = mdp.start_state_function(self.states[state])
