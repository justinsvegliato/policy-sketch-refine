import math
import numpy as np

from random import randint, random

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

MIN_VISIBILITY = 0
MAX_VISIBILITY = 2

DEFAULT_NUM_POI = 5

# TODO: allow single dict to specify everything?
class EarthObservationMDP:
    def __init__(self, size=None, points_of_interest=None, visibility=None):
        if size = None:
            self.size = (12, 12)
        else:
            self.size = size
            
        # dictionary containing location tuple and starting visibility for each POI.
        # poi_descrition = {(x, y):vis, ...}
        self.poi_description = {} 
        
        # Set the points of interest
        self.num_points_of_interest = 0
        if points_of_interest = None:
            self.num_points_of_interest = DEFAULT_NUM_POI
            self.__initRandomPointsOfInterest()
        elif isinstance(points_of_interest, int):
            self.num_points_of_interest = points_of_interest
            self.__initRandomPointsOfInterest()
        elif isinstance(points_of_interest, list):
            self.num_points_of_interest = len(points_of_interest)
            self.__initSetPointsOfInterest(points_of_interest)
        else:
            print("POI arg not parsed correctly")
        
        # Set the visibility
        if visibility = None:
            self.__initRandomVisibility()
        elif isinstance(visibility, int):
            self.__initConstantVisibility(visibility)
        elif isinstance(visibility, list):
            self.__initExactVisibility(visibility)
        else:
            print("Visibility arg not parsed correctly")
        
    def __initRandomPointsOfInterest(self):
        # Random initialization
        while len(self.poi_description) < self.num_points_of_interest:
            rand_lat = randint(0, self.size[0])
            rand_long = randint(0, self.size[1])
            rand_loc = (rand_long, rand_lat)
            self.poi_description[rand_loc] = 0

    def __initSetPointsOfInterest(self, points_of_interest):
        for point in points_of_interest:
            self.poi_description[point] = 0

    
    
    def __initVisibility(self):
        # Random initialization
        self.visibility = np.empty([self.size, self.size], dtype=int)
        for x in range(self.size):
            for y in range(self.size):
                self.visibility[x][y] = randint(0, 10)




    def __weatherEvolution(self):


        
        
        
        
        
        
        
        
        
        
    def states(self):
        return list(range()) #TODO: better way to do this

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


        # TODO: take into account boundary conditions for satellite pos and weather limits...
        #       try to limit weather horizon
        # Weather model: %10 chance +/- 2 vis (5% each), 20% +/- 1 vis (10% each), 70% chance no change
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
            #NOTE: One possible solution is to limit the weather updates to some subset of the "world" relative to the satellite,
            #      say, any cell reachable within k timesteps. This would give roughly k(k+1) + k as the exponent, which is hopefully << 10,000
            return pow(BIG_WEATHER_PROB, big_change) * pow(SMALL_WEATHER_PROB, small_change) * pow(NO_WEATHER_PROB, no_change)


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
