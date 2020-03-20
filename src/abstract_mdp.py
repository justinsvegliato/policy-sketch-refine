import math
import statistics


ABSTRACTION = {
    'MEAN': lambda ground_values, ground_states: sum(ground_values) / float(len(ground_states)),
    'MEDIAN': lambda ground_values, _: statistics.median(ground_values),
    'MIDPOINT': lambda ground_values, _: (min(ground_values) + max(ground_values)) / 2.0
}


class AbstractMDP:
    def __create_new_partition(self, abstract_states, abstract_state):
        ground_states = abstract_states[abstract_state]

        size = len(ground_states)
        abstract_states[abstract_state] = ground_states[0:math.floor(size / 2)]
        abstract_states[len(abstract_states)] = ground_states[math.floor(size / 2):size]

        return abstract_states

    def __check_block_reward_uniformity(self, mdp, abstract_states, abstract_state):
        for action in mdp.actions():
            # TODO Make more efficient by indexing into block states since abs(a - b) == abs(b - a)
            for ground_state in abstract_states[abstract_state]:
                for other_ground_state in abstract_states[abstract_state]:
                    difference = abs(mdp.reward_function(ground_state, action) - mdp.reward_function(other_ground_state, action))
                    if difference > self.epsilon:
                        return False
        return True

    def __check_block_transition_stability(self, mdp, abstract_states, abstract_state):
        for other_abstract_state in abstract_states:
            if other_abstract_state != abstract_state:
                for action in mdp.actions():
                    transition_probabilities = {}
                    for ground_state in abstract_states[abstract_state]:
                        transition_probability = 0.0
                        for successor_ground_state in abstract_states[other_abstract_state]:
                            transition_probability += mdp.transition_function(ground_state, action, successor_ground_state)
                        transition_probabilities[ground_state] = transition_probability

                    # TODO Make more efficient by indexing into block states since abs(a - b) == abs(b - a)
                    for ground_state in abstract_states[abstract_state]:
                        for other_ground_state in abstract_states[abstract_state]:
                            difference = abs(transition_probabilities[ground_state] - transition_probabilities[other_ground_state])
                            if difference > self.epsilon:
                                return False

        return True

    # NOTE Not sure if it's more efficient to check rewards first for all blocks and then
    # transitions for all blocks, or check both for a single block at a time. We're doing
    # the latter currently.
    def __check_stability(self, mdp, abstract_states):
        for abstract_state in abstract_states:
            is_epsilon_uniform = self.__check_block_reward_uniformity(mdp, abstract_states, abstract_state)
            if not is_epsilon_uniform:
                return False, abstract_state

            is_epsilon_stable = self.__check_block_transition_stability(mdp, abstract_states, abstract_state)
            if not is_epsilon_stable:
                return False, abstract_state

        return True, None

    def __compute_abstract_states(self, mdp):
        abstract_states = {0: mdp.states()}
        is_stable, abstract_state = self.__check_stability(mdp, abstract_states)

        while not is_stable:
            abstract_states = self.__create_new_partition(abstract_states, abstract_state)
            is_stable, abstract_state = self.__check_stability(mdp, abstract_states)

        return abstract_states

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

                for successor_abstract_state, successor_ground_states in self.abstract_states.items():
                    ground_transition_probabilities = []
                    for ground_state in ground_states:
                        for successor_ground_state in successor_ground_states:
                            ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, successor_ground_state))

                    abstract_transition_probability = ABSTRACTION[self.abstraction](ground_transition_probabilities, ground_states)
                    abstract_transition_probabilities[abstract_state][abstract_action][successor_abstract_state] = abstract_transition_probability

                    normalizer += abstract_transition_probability

                for successor_abstract_state in self.abstract_states:
                    abstract_transition_probabilities[abstract_state][abstract_action][successor_abstract_state] /= normalizer

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

    def __init__(self, mdp, epsilon, abstraction):
        self.epsilon = epsilon
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError('Invalid parameter provided: epsilon must be between 0 and 1 inclusive')

        self.abstraction = abstraction
        if not self.abstraction in ABSTRACTION:
            raise ValueError(f'Invalid parameter provided: bound type must be in {list(ABSTRACTION)}')

        self.abstract_states = self.__compute_abstract_states(mdp)
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
