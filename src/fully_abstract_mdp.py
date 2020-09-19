import math
import statistics

ABSTRACTION = {
    'MEAN': lambda ground_values, ground_states: sum(ground_values) / float(len(ground_states)),
    'MEDIAN': lambda ground_values, _: statistics.median(ground_values),
    'MIDPOINT': lambda ground_values, _: (min(ground_values) + max(ground_values)) / 2.0,
    'MAX': lambda ground_values, _: max(ground_values)
}


class FullyAbstractMDP:
    # TODO: Clean up parsing logic
    # TODO: Optimize this function - it can be shorter/faster
    # def __is_relevant(self, abstract_state, abstract_successor_state):
    #     abstract_state_id = int(abstract_state.split('_')[1])
    #     abstract_successor_state_id = int(abstract_successor_state.split('_')[1])

    #     # Check if the states are the same
    #     if abstract_state_id == abstract_successor_state_id:
    #         return True

    #     # Check the leftmost column
    #     if abstract_state_id % self.abstract_width == 0:
    #         if abstract_successor_state_id == abstract_state_id + 1:
    #             return True
    #     # Check the rightmost column
    #     elif abstract_state_id % self.abstract_width == self.abstract_width - 1:
    #         if abstract_successor_state_id == abstract_state_id - 1:
    #             return True
    #     # Check some column between the leftmost and rightmost columns
    #     else:
    #         if abstract_successor_state_id in (abstract_state_id - 1, abstract_state_id + 1):
    #             return True

    #     # Check if the abstract successor state is above or below the abstract state
    #     if abstract_successor_state_id in (abstract_state_id - self.abstract_width, abstract_state_id + self.abstract_width):
    #         return True

    #     return False

    def __is_relevant(self, abstract_state, abstract_successor_state):
        return True

    def compute_abstract_states(self, mdp):
        abstract_states = {}

        ground_states = mdp.states()

        reward_states = []
        for ground_state in ground_states:
            reward = mdp.reward_function(ground_state, 'STAY')
            if reward > 0:
                reward_states.append(ground_state)

        for abstract_row_index in range(self.abstract_height):
            for abstract_column_index in range(self.abstract_width):
                block_rows = self.abstract_state_height
                if abstract_row_index == self.abstract_height - 1:
                    block_rows += mdp.height - self.abstract_state_height * (abstract_row_index + 1)

                block_cols = self.abstract_state_width
                if abstract_column_index == self.abstract_width - 1:
                    block_cols += mdp.width - self.abstract_state_width * (abstract_column_index + 1)

                relevant_ground_states = []

                for row_index in range(block_rows):
                    for column_index in range(block_cols):
                        row_offset = abstract_row_index * self.abstract_state_height
                        column_offset = abstract_column_index * self.abstract_state_width

                        ground_state_index = mdp.width * (row_offset + row_index) + (column_offset + column_index)
                        ground_state = ground_states[ground_state_index]

                        if ground_state not in reward_states:
                            relevant_ground_states.append(ground_state)

                abstract_state_index = self.abstract_width * abstract_row_index + abstract_column_index
                abstract_states['abstract_%s' % abstract_state_index] = relevant_ground_states

        abstract_states['abstract_%s' % len(abstract_states)] = reward_states

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

                for abstract_successor_state, ground_successor_states in self.abstract_states.items():
                    if not self.__is_relevant(abstract_state, abstract_successor_state):
                        abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = 0
                        continue

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

    def __init__(self, mdp, abstraction, abstract_state_width, abstract_state_height):
        self.abstraction = abstraction
        if not self.abstraction in ABSTRACTION:
            raise ValueError(f"Invalid parameter provided: abstraction must be in {list(ABSTRACTION)}")

        self.abstract_state_width = abstract_state_width
        self.abstract_state_height = abstract_state_height
        if not self.abstract_state_width > 0 or not self.abstract_state_height > 0:
            raise ValueError(f"Invalid parameters provided: abstract_state_height and abstract_state_width must be greater than 0")

        self.abstract_width = math.ceil(mdp.width / self.abstract_state_width)
        self.abstract_height = math.ceil(mdp.height / self.abstract_state_height)

        self.abstract_states = self.compute_abstract_states(mdp)
        print(self.abstract_states)
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
            ground_states = ground_states + self.abstract_states[abstract_state]
        return ground_states
