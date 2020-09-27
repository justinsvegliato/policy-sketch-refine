import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import printer
import utils

ABSTRACTION = {
    'MEAN': lambda ground_values, ground_states: sum(ground_values) / float(len(ground_states)),
    'MAX': lambda ground_values, _: max(ground_values)
}

GS_SAMPLES = 67 
GSS_SAMPLES = 5 
NUM_PROCESSES = 4


def task(mdp, state_space, abstract_mdp):
    results = {}

    for abstract_state, ground_states in state_space:
        results[abstract_state] = {}
        abstract_state_index = int((abstract_state.split("_"))[1])
        abstract_weather_index = int((abstract_state.split("_"))[2])

        for abstract_action in abstract_mdp.abstract_actions:
            results[abstract_state][abstract_action] = {}

            normalizer = 0

            for abstract_successor_state, ground_successor_states in abstract_mdp.abstract_states.items():
                abstract_successor_state_index = int((abstract_successor_state.split("_"))[1])
                abstract_successor_weather_index = int((abstract_successor_state.split("_"))[2])

                # Satellite can only end up in 6 abstract states - the current, upper, lower, right, right upper, right lower.
                # There are some conditions with even fewer possibilities.
                # These are further limited by the choice of action.
                abstract_state_col = abstract_state_index % abstract_mdp.abstract_mdp_width
                abstract_state_row = math.floor(abstract_state_index / abstract_mdp.abstract_mdp_width)
                abstract_successor_state_col = abstract_successor_state_index % abstract_mdp.abstract_mdp_width
                abstract_successor_state_row = math.floor(abstract_successor_state_index / abstract_mdp.abstract_mdp_width)

                is_possible_successor = True

                # STAY and IMAGE cannot shift focus North or South
                if (abstract_action == 'STAY' or abstract_action == 'IMAGE') and (abstract_state_row != abstract_successor_state_row):
                    is_possible_successor = False

                # SOUTH cannot shift focus North
                if abstract_action == 'SOUTH' and abstract_state_row != abstract_successor_state_row and abstract_successor_state_row != abstract_state_row + 1:
                    is_possible_successor = False

                # NORTH cannot shift focus South
                if abstract_action == 'NORTH' and abstract_state_row != abstract_successor_state_row and abstract_successor_state_row != abstract_state_row - 1:
                    is_possible_successor = False

                # If on the far east column
                if abstract_state_col == abstract_mdp.abstract_mdp_width - 1:
                    if abstract_state_col != abstract_successor_state_col and abstract_successor_state_col != 0:
                        is_possible_successor = False

                # If not on the far east column
                else:
                    if abstract_state_col != abstract_successor_state_col and abstract_successor_state_col != abstract_state_col + 1:
                        is_possible_successor = False

                if is_possible_successor:
                    ground_transition_probabilities = []
                    abstract_transition_probability = None

                    if GS_SAMPLES:
                        sampled_ground_states = np.random.choice(ground_states, GS_SAMPLES, replace=False)
                        for ground_state in sampled_ground_states:
                            #likely_successor_ground_states = mdp.get_successors(ground_state, abstract_action)
                            #relevant_successor_ground_states = list(likely_successor_ground_states.intersection(set(ground_successor_states)))
                            #if len(relevant_successor_ground_states) == 0:
                            #    #print("zero relevant succ")
                            #    continue
                            #sampled_ground_successor_states = np.random.choice(relevant_successor_ground_states, GSS_SAMPLES)

                            #for ground_successor_state in sampled_ground_successor_states:
                            #    ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, ground_successor_state))


                            likely_ground_successor_states = mdp.get_successors(ground_state, abstract_action)
                            for ground_successor_state in ground_successor_states:
                                if ground_successor_state in likely_ground_successor_states:
                                    ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, ground_successor_state))


                        abstract_transition_probability = ABSTRACTION[abstract_mdp.abstraction](ground_transition_probabilities, sampled_ground_states)
                    else:
                        for ground_state in ground_states:
                            visibility_fidelity = mdp.get_visibility_fidelity()
                            assert(visibility_fidelity > 1)
                            lower_vis = math.floor(visibility_fidelity / 2) - 1 # At or below is considered poor vis
                            upper_vis = lower_vis + 1 # At or above is considered good vis

                            # Assume min visibility = 0
                            weather_partition = [range(0, lower_vis + 1), range(upper_vis, visibility_fidelity)]
                            _, ground_weather_status = mdp.get_state_factors_from_state(ground_state)

                            locations = sorted(list(ground_weather_status.keys()))

                            extreme_ground_weather = []
                            for loc in locations:
                                if ground_weather_status[loc] + 1 in weather_partition[0]:
                                    extreme_ground_weather.append(-1)
                                elif ground_weather_status[loc] - 1 in weather_partition[1]:
                                    extreme_ground_weather.append(1)
                                else:
                                    extreme_ground_weather.append(0)

                            extreme_abstract_weather = []
                            num_points_of_interest = mdp.get_num_point_of_interests()
                            partial_weather_partition_status = abstract_successor_weather_index
                            for location_index in range(num_points_of_interest - 1, -1, -1):
                                # When location index is high, ids are more contiguous - this is how we match to the ground state definitions
                                location_divisor = pow(2, location_index)

                                # Location i has lower_vis or less visibility
                                if (math.floor(partial_weather_partition_status / location_divisor < 1)):
                                    extreme_abstract_weather.insert(0, -1)

                                # Location i has upper_vis or greater visibility
                                elif (math.floor(partial_weather_partition_status / location_divisor < 2)):
                                    extreme_abstract_weather.insert(0, 1)

                                partial_weather_partition_status = partial_weather_partition_status % location_divisor

                            # If every weather has a chance of transitioning to a weather in the abstract successor state
                            ground_weather_bounds = np.array(extreme_ground_weather)
                            abstract_weather_bounds = np.array(extreme_abstract_weather)
                            if 2 not in np.absolute(ground_weather_bounds - abstract_weather_bounds):
                                likely_ground_successor_states = mdp.get_successors(ground_state, abstract_action)
                                for ground_successor_state in ground_successor_states:
                                    if ground_successor_state in likely_ground_successor_states:
                                        ground_transition_probabilities.append(mdp.transition_function(ground_state, abstract_action, ground_successor_state))

                        abstract_transition_probability = ABSTRACTION[abstract_mdp.abstraction](ground_transition_probabilities, ground_states)

                    results[abstract_state][abstract_action][abstract_successor_state] = abstract_transition_probability
                    normalizer += abstract_transition_probability
                else:
                    results[abstract_state][abstract_action][abstract_successor_state] = 0.0

            for abstract_successor_state in abstract_mdp.abstract_states:
                results[abstract_state][abstract_action][abstract_successor_state] /= normalizer

    return results


class EarthObservationAbstractMDP:
    # TODO: Test and polish this function
    def compute_abstract_states(self, mdp):
        abstract_states = {}

        num_points_of_interest = mdp.get_num_point_of_interests()
        visibility_fidelity = mdp.get_visibility_fidelity()
        weather_expansion_factor = pow(visibility_fidelity, num_points_of_interest)
        
        assert(visibility_fidelity > 1)
        lower_vis = math.floor(visibility_fidelity / 2) - 1 # At or below is considered poor vis
        upper_vis = lower_vis + 1 # At or above is considered good vis

        # Assume min visibility = 0
        weather_partition = [range(0, lower_vis + 1), range(upper_vis, visibility_fidelity)]

        # Number of abstract states which have identical location, but different weather
        num_weather_conditions = pow(2, num_points_of_interest)

        for abstract_row_index in range(self.abstract_mdp_height):
            for abstract_column_index in range(self.abstract_mdp_width):
                block_rows = self.abstract_state_height
                if abstract_row_index == self.abstract_mdp_height - 1:
                    block_rows += mdp.height() - self.abstract_state_height * (abstract_row_index + 1)

                block_cols = self.abstract_state_width
                if abstract_column_index == self.abstract_mdp_width - 1:
                    block_cols += mdp.width() - self.abstract_state_width * (abstract_column_index + 1)

                for weather_partition_status in range(0, num_weather_conditions):
                    abstract_state_index = self.abstract_mdp_width * abstract_row_index + abstract_column_index
                    abstract_states[f'abstract_{abstract_state_index}_{weather_partition_status}'] = []

                    for row_index in range(block_rows):
                        for column_index in range(block_cols):
                            row_offset = abstract_row_index * self.abstract_state_height
                            column_offset = abstract_column_index * self.abstract_state_width

                            partial_weather_partition_status = weather_partition_status
                            eligible_weather_states = range(0, weather_expansion_factor)
                            for location_index in range(num_points_of_interest - 1, -1, -1):
                                
                                # When location index is high, ids are more contiguous - this is how we match to the ground state definitions
                                location_divisor = pow(2, location_index)
                                
                                # Location i has lower_vis or less visibility
                                if (math.floor(partial_weather_partition_status / location_divisor < 1)):
                                    eligible_weather_states = [x for x in eligible_weather_states if math.floor((x % pow(visibility_fidelity, location_index + 1)) / pow(visibility_fidelity, location_index)) in weather_partition[0]]

                                # Location i has upper_vis or greater visibility
                                elif (math.floor(partial_weather_partition_status / location_divisor < 2)):
                                    eligible_weather_states = [x for x in eligible_weather_states if math.floor((x % pow(visibility_fidelity, location_index + 1)) / pow(visibility_fidelity, location_index)) in weather_partition[1]]
                                else:
                                    assert "sum ting wong"

                                partial_weather_partition_status = partial_weather_partition_status % location_divisor

                            ground_state_anchor_index = weather_expansion_factor * (mdp.width() * (row_offset + row_index) + (column_offset + column_index))
                            ground_states = [x + ground_state_anchor_index for x in eligible_weather_states]
                            abstract_states[f'abstract_{abstract_state_index}_{weather_partition_status}'] += ground_states

        return abstract_states

    def __compute_abstract_rewards(self, mdp):
        abstract_rewards = {}

        statistics = {'count': 0, 'total': len(self.abstract_states) * len(self.abstract_actions)}

        for abstract_state, ground_states in self.abstract_states.items():
            abstract_rewards[abstract_state] = {}
            for abstract_action in self.abstract_actions:
                printer.print_loading_bar(statistics['count'], statistics['total'], 'Abstract Rewards')
                statistics['count'] += 1

                ground_rewards = [mdp.reward_function(ground_state, abstract_action) for ground_state in ground_states]
                abstract_reward = ABSTRACTION[self.abstraction](ground_rewards, ground_states)
                abstract_rewards[abstract_state][abstract_action] = abstract_reward

        return abstract_rewards

    def __compute_abstract_transition_probabilities(self, mdp):
        abstract_transition_probabilities = {}

        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as pool:
            partition_futures = []
            state_space_partitions = utils.get_partitions(list(self.abstract_states.items()), NUM_PROCESSES)

            statistics = {'count': 0, 'total': len(state_space_partitions)}

            for state_space in state_space_partitions:
                printer.print_loading_bar(statistics['count'], statistics['total'], "Abstract Transition Probabilities")
                statistics['count'] += 1

                partition_future = pool.submit(task, mdp, state_space, self)
                partition_futures.append(partition_future)

            for partition_future in partition_futures:
                result = partition_future.result()
                for key in result:
                    abstract_transition_probabilities[key] = result[key]

        return abstract_transition_probabilities

    def __compute_abstract_start_state_probabilities(self, mdp):
        abstract_start_state_probabilities = {}

        statistics = {'count': 0, 'total': len(self.abstract_states)}

        normalizer = 0

        for abstract_state, ground_states in self.abstract_states.items():
            printer.print_loading_bar(statistics['count'], statistics['total'], 'Abstract Start State Probabilities')
            statistics['count'] += 1

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

        self.abstract_state_width = abstract_state_width
        self.abstract_state_height = abstract_state_height

        self.abstract_mdp_width = math.ceil(mdp.width() / self.abstract_state_width)
        self.abstract_mdp_height = math.ceil(mdp.height() / self.abstract_state_height)

        self.ground_states = {}

        # NOTE: You can use the basic trans probs with either abstract state space representation. However, you 
        # must use the regular (includes weather) abstraction when using the regular transition function
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
        if ground_state not in self.ground_states:
            for abstract_state, ground_states in self.abstract_states.items():
                if ground_state in ground_states:
                    self.ground_states[ground_state] = abstract_state
                    return abstract_state 
   
        return self.ground_states[ground_state]

    def get_ground_states(self, abstract_states):
        ground_states = []
        for abstract_state in abstract_states:
            ground_states = ground_states + self.abstract_states[abstract_state]
        return ground_states
