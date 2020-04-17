import math
import numpy as np

BIG_WEATHER_PROB = 0.05
SMALL_WEATHER_PROB = 0.1
NO_WEATHER_PROB = 0.7

ACTION_DETAILS = {
    'STAY': {
    },
    'NORTH': {
    },
    'SOUTH': {
    },
    'IMAGE': {
    },
}

class EarthObservationMDP:
    def __init__(self, size):
        self.size = size
        self.initVisibility()
        self.initPointsOfInterest()

    def __initVisibility(self):
        self.visibility = np.empty([self.size, self.size], dtype=int)
        #TODO: finish init / maybe should be argument

    def __initPointsOfInterest(self):
        self.points_of_interest = np.empty([self.size, self.size], dtype=bool)
        #TODO: finish init / maybe should be argument
    
    def __weatherEvolution(self):

    def states(self):
        return list(range(2200000000)) #TODO: better way to do this

    def actions(self):
        return list(ACTION_DETAILS.keys())

    def transition_function(self, state, action, successor_state):

        # Periodic boundaries for East-West direction
        if state.col == self.size - 1 and successor_state.col != 0:
            return 0
        
        # We always move East by one grid cell
        if state.col != successor_state.col - 1:
            return 0

        # STAY and IMAGE cannot shift focus North-South
        if (ACTION_DETAILS[action] == 'STAY' or ACTION_DETAILS[action] == 'IMAGE') and state.row != successor_state.row:
            return 0

        # At the bottom, SOUTH does nothing
        if ACTION_DETAILS[action] == 'SOUTH' and state.row == self.size - 1 and state.row != successor_state.row:
            return 0

        # Otherwise, it always goes south by one cell
        if ACTION_DETAILS[action] == 'SOUTH' and state.row != successor_state.row + 1:
            return 0

        # At the top, NORTH does nothing
        if ACTION_DETAILS[action] == 'NORTH' and state.row == 0 and state.row != successor_state.row:
            return 0

        # Otherwise, it always goes north by one cell
        if ACTION_DETAILS[action] == 'NORTH' and state.row != successor_state.row - 1:
            return 0

        # Weather model: %10 chance +/- 2 vis (5 each), 20% +/- 1 vis (10 each), 70% chance no change
        weather_diff = np.absolute(state.visibility - successor_state.visibility)
        if np.any(weather_diff > 2): 
            return 0
        else:
            big_change = np.count_nonzero(weather_diff == 2)
            small_change = np.count_nonzero(weather_diff == 1)
            no_change = np.count_nonzero(weather_diff == 0)
            #NOTE: I'm concerned that this number will be far too small - it's maximum value is 0.7^(10,000), or (NO_WEATHER_PROB ^ (size * size)).
            #NOTE: on the bright side, it's possible using sketch refine will fix this problem to some degree since the probabilities 
            #      in the abstract MDP's transition function should be much higher
            return (big_change * BIG_WEATHER_PROB) * (small_change * SMALL_WEATHER_PROB) * (no_change * NO_WEATHER_PROB)

        print("I reached the end and didn't find a return... something is broken in EarthObservationMDP transition function.")
        return 0

    def reward_function(self, state, action):

        if state.is_poi[state.row][state.col] and ACTION_DETAILS[action] == 'IMAGE':
            return 0.1 * state.visibility[state.row][state.col] 
        elif not state.is_poi[state.row][state.col] and ACTION_DETAILS[action] == 'IMAGE':
            return -0.01

        return 0

    #TODO: need to update this so it works with state representation
    def start_state_function(self, state):
        start_states = []

        for row in range(self.height):
            for column in range(self.width):
                if self.grid_world[row][column] != 'W':
                    start_states.append(self.width * row + column)

        return 1.0 / len(start_states) if state in start_states else 0
