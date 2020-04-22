import statistics


ABSTRACTION = {
    'MEAN': lambda ground_values, ground_states: sum(ground_values) / float(len(ground_states)),
    'MEDIAN': lambda ground_values, _: statistics.median(ground_values),
    'MIDPOINT': lambda ground_values, _: (min(ground_values) + max(ground_values)) / 2.0
}


class AbstractMDP:
    def __compute_abstract_rewards(self, mdp):
        abstract_rewards = {}

        for abstract_state, ground_states in self.abstract_states.items():
            abstract_rewards[abstract_state] = {}

            for abstract_action in self.abstract_actions:
                ground_rewards = [mdp.reward_function(ground_state, abstract_action) for ground_state in ground_states]

                abstract_reward = ABSTRACTION[self.abstraction](ground_rewards, ground_states)
                abstract_rewards[abstract_state][abstract_action] = abstract_reward

        return abstract_rewards

    def __compute_abstract_transition_probabilities(self, mdp):
        abstract_transition_probabilities = {}

        for abstract_state, ground_states in self.abstract_states.items():
            abstract_transition_probabilities[abstract_state] = {}

            for abstract_action in self.abstract_actions:
                abstract_transition_probabilities[abstract_state][abstract_action] = {}

                normalizer = 0

                for abstract_successor_state, ground_successor_states in self.abstract_states.items():
                    ground_transition_probabilities = []
                    for ground_state in ground_states:
                        for successor_ground_state in ground_successor_states:
                            ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, successor_ground_state))

                    abstract_transition_probability = ABSTRACTION[self.abstraction](ground_transition_probabilities, ground_states)
                    abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = abstract_transition_probability

                    normalizer += abstract_transition_probability

                for abstract_successor_state in self.abstract_states:
                    abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] /= normalizer

        return abstract_transition_probabilities

    def __compute_abstract_start_state_probabilities(self, mdp):
        abstract_start_state_probabilities = {}

        normalizer = 0

        for abstract_state, ground_states in self.abstract_states.items():
            ground_start_state_probabilities = [mdp.start_state_function(ground_state) for ground_state in ground_states]

            abstract_start_state_probability = ABSTRACTION[self.abstraction](ground_start_state_probabilities, ground_states)
            abstract_start_state_probabilities[abstract_state] = abstract_start_state_probability

            normalizer += abstract_start_state_probability

        for abstract_state in self.abstract_states:
            abstract_start_state_probabilities[abstract_state] /= normalizer

        return abstract_start_state_probabilities

    def __init__(self, mdp, abstraction):
        self.abstraction = abstraction
        if not self.abstraction in ABSTRACTION:
            raise ValueError(f"Invalid parameter provided: abstraction must be in {list(ABSTRACTION)}")

        self.abstract_states = mdp.compute_abstract_states(mdp)
        self.abstract_actions = mdp.actions()
        self.abstract_rewards = self.__compute_abstract_rewards(mdp)
        self.abstract_transition_probabilities = self.__compute_abstract_transition_probabilities(mdp)
        self.abstract_start_state_probabilities = self.__compute_abstract_start_state_probabilities(mdp)

    def states(self):
        return list(self.abstract_states)

    def actions(self):
        return self.abstract_actions

    def transition_function(self, state, action, successor_state):
        return self.abstract_transition_probabilities[state][action][successor_state]

    def reward_function(self, state, action):
        return self.abstract_rewards[state][action]

    def start_state_function(self, state):
        return self.abstract_start_state_probabilities[state]

    def get_abstract_state(self, ground_state):
        for abstract_state, ground_states in self.abstract_states.items():
            if ground_state in ground_states:
                return abstract_state
        return None

    def get_ground_states(self, abstract_states):
        ground_states = []
        for abstract_state in abstract_states:
            some_ground_states = self.abstract_states[abstract_state]
            for ground_state in some_ground_states:
                ground_states.append(ground_state)
        return ground_states
