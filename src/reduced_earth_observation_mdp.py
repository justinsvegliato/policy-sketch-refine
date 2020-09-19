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

PROB_WEATHER_GETS_WORSE = 0.1
PROB_WEATHER_GETS_BETTER = 0.1
PROB_WEATHER_STAYS_SAME = 0.8

MIN_VISIBILITY = 0
MAX_VISIBILITY = 2

DEFAULT_NUM_POI = 5

class ReducedEarthObservationMDP:
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
        elif isinstance(visibility, dict):
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

    def __initRandomVisibility(self):
        # Random initialization
        for point in self.poi_description:
            rand_vis = randint(MIN_VISIBILITY, MAX_VISIBILITY)
            self.poi_description[point] = rand_vis
    
    def __initConstantVisibility(self, visibility):
        # Uniform visibility
        visibility = max(MIN_VISIBILITY, visibility)
        visibility = min(MAX_VISIBILITY, visibility)
        for point in self.poi_description:
            self.poi_description[point] = visibility
    
    def __initExactVisibility(self, visibility):
        self.poi_description = visibility

    def __weatherEvolution(self):
        # Not sure if we need this....
        placeholder = 1
        
    def states(self):
        locations = self.size[0] * self.size[1]
        base = (MAX_VISIBILITY - MIN_VISIBILITY + 1)
        power = self.num_points_of_interest
        total_number_of_states = pow(base, power) * locations
        return list(range(total_number_of_states)) 

    def actions(self):
        return list(ACTION_DETAILS.keys())

    def transition_function(self, state, action, successor_state):

        # Periodic boundaries for East-West direction
        if state.col == self.size[1] - 1 and successor_state.col != 0:
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

        weather_transition_prob = 1.0
        for loc in state.poi_weather:
            current_loc_weather = state.poi_weather[loc]
            new_loc_weather = successor_state.poi_weather[loc]

            assert (current_loc_weather >= MIN_VISIBILITY and current_loc_weather <= MAX_VISIBILITY),"Bad weather value in current state"
            assert (new_loc_weather >= MIN_VISIBILITY and new_loc_weather <= MAX_VISIBILITY),"Bad weather value in successor state"

            # Weather in our model cannot change from good to bad immediately
            if abs(new_loc_weather - current_loc_weather) > 1:
                return 0

            elif current_loc_weather == MIN_VISIBILITY:
                # Weather cannot get worse than minimum visibility
                if new_loc_weather == current_loc_weather:
                    weather_transition_prob *= (PROB_WEATHER_GETS_WORSE + PROB_WEATHER_STAYS_SAME)
                else:
                    weather_transition_prob *= PROB_WEATHER_GETS_BETTER
            
            elif current_loc_weather == MAX_VISIBILITY:
                # Weather cannot get better than maximum visibility
                if new_loc_weather == current_loc_weather:
                    weather_transition_prob *= (PROB_WEATHER_GETS_BETTER + PROB_WEATHER_STAYS_SAME)
                else:
                    weather_transition_prob *= PROB_WEATHER_GETS_WORSE
                
            else:
                if new_loc_weather == current_loc_weather:
                    weather_transition_prob *= PROB_WEATHER_STAYS_SAME
                elif new_loc_weather > current_loc_weather:
                    weather_transition_prob *= PROB_WEATHER_GETS_BETTER
                else:
                    weather_transition_prob *= PROB_WEATHER_GETS_WORSE
                
    def reward_function(self, state, action):
        if state.loc in state.poi_weather and ACTION_DETAILS[action] == 'IMAGE':
            return 1.0 + 1.0 * state.poi_weather[state.loc]
        return -0.01

    def start_state_function(self, state):
        return 1.0 / len(self.states()) 
