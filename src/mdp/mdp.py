# Memory representation of an MDP


class MDP(object):
    def __init__(self):
        self.n_states = None
        self.n_actions = None
        self.start_probability = None
        self.rewards = None
        self.transitions = None

    def load_dummy(self, n_states, n_actions):
        """
        Load a dummy MDP (for debugging).
        """
        import numpy as np
        np.random.seed(0)

        self.n_states = n_states
        self.n_actions = n_actions

        # Start probability
        self.start_probability = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

        # Random rewards
        self.rewards = np.random.randint(0, 10, size=(self.n_states, self.n_actions))

        # Random transitions
        self.transitions = np.empty(shape=(self.n_states, self.n_actions, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                self.transitions[i, j] = np.random.dirichlet(np.ones(self.n_states), size=1)[0]

        # print(self.rewards)
        # # print(self.transitions)
        # for i in range(self.n_states):
        #     for j in range(self.n_actions):
        #         for k in range(self.n_states):
        #             print(i, j, k, self.transitions[i, j, k])

    def load_from_csv(self):
        pass

    def formulate_lp(self):
        assert self.start_probability is not None
        assert self.rewards is not None
        assert self.transitions is not None
        assert self.start_probability.shape == (self.n_states,)
        assert self.rewards.shape == (self.n_states, self.n_actions)
        assert self.transitions.shape == (self.n_states, self.n_actions, self.n_states)

    def solve_lp(self):
        raise NotImplementedError
