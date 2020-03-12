import numpy as np

class MemoryMDP(object):
    def __init__(self):
        self.n_states = None
        self.n_actions = None
        self.rewards = None
        self.transition_probabilities = None
        self.start_probabilities = None

    def load_mdp(self, mdp):
        states = mdp.states()
        actions = mdp.actions()

        self.n_states = len(states)
        self.n_actions = len(actions)

        self.rewards = np.zeros(shape=(self.n_states, self.n_actions))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                self.rewards[i, j] = mdp.reward_function(states[i], actions[j])

        self.transition_probabilities = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    self.transition_probabilities[i, j, k] = mdp.transition_function(states[i], actions[j], states[k])

        self.start_probabilities = np.zeros(self.n_states)
        for i in range(self.n_states):
            self.start_probabilities[i] = mdp.start_state_function(states[i])

    def load_random_mdp(self, n_states, n_actions):
        np.random.seed(0)

        self.n_states = n_states
        self.n_actions = n_actions

        self.rewards = np.random.randint(0, 10, size=(self.n_states, self.n_actions))

        self.transition_probabilities = np.empty(shape=(self.n_states, self.n_actions, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                self.transition_probabilities[i, j] = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

        self.start_probabilities = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

    def formulate_lp(self):
        assert self.n_states is not None
        assert self.n_actions is not None
        assert self.rewards is not None
        assert self.transition_probabilities is not None
        assert self.start_probabilities is not None
        assert self.rewards.shape == (self.n_states, self.n_actions)
        assert self.transition_probabilities.shape == (self.n_states, self.n_actions, self.n_states)
        assert self.start_probabilities.shape == (self.n_states,)

    def solve_lp(self):
        raise NotImplementedError
