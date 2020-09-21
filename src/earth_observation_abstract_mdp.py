import logging
import math

import numpy as np

import printer

ABSTRACTION = {
    'MEAN': lambda ground_values, ground_states: sum(ground_values) / float(len(ground_states)),
    'MAX': lambda ground_values, _: max(ground_values)
}

SAMPLES = None

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-15s|%(levelname)-4s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


class EarthObservationAbstractMDP:
    def compute_abstract_states(self, mdp):
        abstract_states = {}

        num_points_of_interest, visual_fidelity = mdp.get_num_POI_num_vis()
        weather_expansion_factor = pow(visual_fidelity, num_points_of_interest)

        for abstract_row_index in range(self.abstract_mdp_height):
            for abstract_column_index in range(self.abstract_mdp_width):
                block_rows = self.abstract_state_height
                if abstract_row_index == self.abstract_mdp_height - 1:
                    block_rows += mdp.height() - self.abstract_state_height * (abstract_row_index + 1)

                block_cols = self.abstract_state_width
                if abstract_column_index == self.abstract_mdp_width - 1:
                    block_cols += mdp.width() - self.abstract_state_width * (abstract_column_index + 1)

                abstract_state_index = self.abstract_mdp_width * abstract_row_index + abstract_column_index
                abstract_states[f'abstract_{abstract_state_index}'] = []

                for row_index in range(block_rows):
                    for column_index in range(block_cols):
                        row_offset = abstract_row_index * self.abstract_state_height
                        column_offset = abstract_column_index * self.abstract_state_width

                        ground_state_anchor_index = weather_expansion_factor * (mdp.width() * (row_offset + row_index) + (column_offset + column_index))
                        
                        abstract_state_index = self.abstract_mdp_width * abstract_row_index + abstract_column_index
                        abstract_states[f'abstract_{abstract_state_index}'] += range(ground_state_anchor_index, ground_state_anchor_index + weather_expansion_factor)

        return abstract_states

    def __compute_abstract_rewards(self, mdp):
        abstract_rewards = {}

        statistics = {
            'count': 0,
            'total': len(self.abstract_states) * len(self.abstract_actions)
        }

        for abstract_state, ground_states in self.abstract_states.items():
            abstract_rewards[abstract_state] = {}
            for abstract_action in self.abstract_actions:
                printer.print_loading_bar(statistics['count'], statistics['total'], 'Abstract Rewards')

                ground_rewards = [mdp.reward_function(ground_state, abstract_action) for ground_state in ground_states]
                abstract_reward = ABSTRACTION[self.abstraction](ground_rewards, ground_states)
                abstract_rewards[abstract_state][abstract_action] = abstract_reward

                statistics['count'] += 1

        return abstract_rewards

    def __compute_abstract_transition_probabilities(self, mdp):
        abstract_transition_probabilities = {}

        statistics = {
            'count': 0,
            'total': len(self.abstract_states) * len(self.abstract_actions) * len(self.abstract_states)
        }

        for abstract_state, ground_states in self.abstract_states.items():
            abstract_transition_probabilities[abstract_state] = {}
            abstract_state_index = int((abstract_state.split("_"))[1])

            for abstract_action in self.abstract_actions:
                abstract_transition_probabilities[abstract_state][abstract_action] = {}

                normalizer = 0

                for abstract_successor_state, ground_successor_states in self.abstract_states.items():
                    printer.print_loading_bar(statistics['count'], statistics['total'], 'Abstract Transition Probabilities')

                    abstract_successor_state_index = int((abstract_successor_state.split("_"))[1])

                    # Satellite can only end up in 6 abstract states - the current, upper, lower, right, right upper, right lower.
                    # There are some conditions with even fewer possibilities.
                    # These are further limited by the choice of action.
                    abstract_state_col = abstract_state_index % self.abstract_mdp_width
                    abstract_state_row = math.floor(abstract_state_index / self.abstract_mdp_width)
                    abstract_successor_state_col = abstract_successor_state_index % self.abstract_mdp_width
                    abstract_successor_state_row = math.floor(abstract_successor_state_index / self.abstract_mdp_width)

                    is_possible_successor = True

                    if (abstract_action == 'IMAGE'):
                        is_possible_successor = False
                        # Necessary to avoid division by zero later, since the rest of this logic is skipped for 'IMAGE' actions
                        normalizer = 1.0
                        abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = \
                                abstract_transition_probabilities[abstract_state]['STAY'][abstract_successor_state]

                    # # STAY and IMAGE cannot shift focus North or South
                    if (abstract_action == 'STAY' or abstract_action == 'IMAGE') and (abstract_state_row != abstract_successor_state_row):
                        is_possible_successor = False
 
                    # SOUTH cannot shift focus North
                    if abstract_action == 'SOUTH' and not ((abstract_state_row != abstract_successor_state_row) or (abstract_state_row != abstract_successor_state_row + 1)):
                        is_possible_successor = False
 
                    # NORTH cannot shift focus South
                    if abstract_action == 'NORTH' and not ((abstract_state_row != abstract_successor_state_row) or (abstract_state_row != abstract_successor_state_row - 1)):
                        is_possible_successor = False

                    # # Not on the far east column
                    if abstract_state_col == self.abstract_mdp_width - 1:
                        if (abstract_state_col != abstract_successor_state_col) and (abstract_successor_state_col != 0):
                            is_possible_successor = False
                    else:
                        if (abstract_state_col != abstract_successor_state_col) and (abstract_state_col != abstract_successor_state_col - 1):
                            is_possible_successor = False

                    if is_possible_successor:
                        ground_transition_probabilities = []

                        if SAMPLES:
                            sampled_ground_states = np.random.choice(ground_states, SAMPLES, replace=False)

                            for ground_state in sampled_ground_states:
                                sampled_ground_successor_states = np.random.choice(ground_successor_states, SAMPLES, replace=False)
                                for ground_successor_state in sampled_ground_successor_states:
                                    ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, ground_successor_state))
                        else:
                            for ground_state in ground_states:
                                for ground_successor_state in ground_successor_states:
                                    ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, ground_successor_state))

                        abstract_transition_probability = ABSTRACTION[self.abstraction](ground_transition_probabilities, ground_states)
                        abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = abstract_transition_probability

                        normalizer += abstract_transition_probability
                    else:
                        abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = 0.0

                    statistics['count'] += 1

                for abstract_successor_state in self.abstract_states:
                    abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] /= normalizer

        return abstract_transition_probabilities

    def __compute_abstract_start_state_probabilities(self, mdp):
        abstract_start_state_probabilities = {}

        statistics = {
            'count': 0,
            'total': len(self.abstract_states)
        }

        normalizer = 0

        for abstract_state, ground_states in self.abstract_states.items():
            printer.print_loading_bar(statistics['count'], statistics['total'], 'Abstract Start State Probabilities')

            ground_start_state_probabilities = [mdp.start_state_function(ground_state) for ground_state in ground_states]

            abstract_start_state_probability = ABSTRACTION[self.abstraction](ground_start_state_probabilities, ground_states)
            abstract_start_state_probabilities[abstract_state] = abstract_start_state_probability

            normalizer += abstract_start_state_probability

            statistics['count'] += 1

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

        self.abstract_state_width = abstract_state_width
        self.abstract_state_height = abstract_state_height

        self.abstract_mdp_width = math.ceil(mdp.width() / self.abstract_state_width)
        self.abstract_mdp_height = math.ceil(mdp.height() / self.abstract_state_height)

        self.abstract_states = self.compute_abstract_states(mdp)
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
