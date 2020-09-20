import copy
import math
import numpy as np

from random import randint, random

ACTION_DETAILS = {
    'STAY': {},
    'NORTH': {},
    'SOUTH': {},
    'IMAGE': {},
}

PROB_WEATHER_GETS_WORSE = 0.1
PROB_WEATHER_GETS_BETTER = 0.1
PROB_WEATHER_STAYS_SAME = 0.8

MIN_VISIBILITY = 0
MAX_VISIBILITY = 2

DEFAULT_NUM_POI = 5

class ReducedEarthObservationMDP:
    def __init__(self, size=None, points_of_interest=None, visibility=None):
        if size == None:
            self.size = (12, 12)
        else:
            self.size = size
            
        # dictionary containing location tuple and starting visibility for each POI.
        # poi_descrition = {(x, y):vis, ...}
        self.poi_description = {} 
        
        # Set the points of interest
        self.num_points_of_interest = 0
        if points_of_interest == None:
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
        if visibility == None:
            self.__initRandomVisibility()
        elif isinstance(visibility, int):
            self.__initConstantVisibility(visibility)
        elif isinstance(visibility, dict):
            self.__initExactVisibility(visibility)
        else:
            print("Visibility arg not parsed correctly")

        self.visibility_fidelity = MAX_VISIBILITY - MIN_VISIBILITY + 1
        
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

    def __stateFactorsFromInt(self, state_id):
        rows = self.size[0]
        cols = self.size[1]
        locations = rows * cols
        base = (MAX_VISIBILITY - MIN_VISIBILITY + 1)
        power = self.num_points_of_interest
        
        # indexing location
        loc_id = math.floor(state_id / pow(base, power))
        latitude = math.floor(loc_id / cols)
        longitude = loc_id % rows
        location = (longitude, latitude)

        # indexing weather
        poi_weather = copy.deepcopy(self.poi_description)
        locs = list(poi_weather.keys())
        assert(power == len(poi_weather)), "inconsistent number of points of interest"
        # Now, overwrite values with whatever the state_id is representing
        weather_id = state_id % pow(base, power)
        for i in range(power-1, -1, -1):
            weather_at_loc = math.floor(weather_id / pow(base, i))
            poi_weather[locs[i]] = weather_at_loc
            weather_id = weather_id % pow(base, i)

        return location, poi_weather

    def getNumPOINumVis(self):
        return self.num_points_of_interest, self.visibility_fidelity

    def getPOILocations(self):
        return list(self.poi_description.keys())

    def width(self):
        return self.size[1]

    def height(self):
        return self.size[0]

    def states(self):
        locations = self.size[0] * self.size[1]
        base = (MAX_VISIBILITY - MIN_VISIBILITY + 1)
        power = self.num_points_of_interest
        total_number_of_states = pow(base, power) * locations
        return list(range(total_number_of_states)) 

    def actions(self):
        return list(ACTION_DETAILS.keys())

    def transition_function(self, state, action, successor_state):
        curr_state_loc, curr_state_weather = self.__stateFactorsFromInt(state)
        successor_state_loc, successor_state_weather = self.__stateFactorsFromInt(successor_state)

        # We always move East by one grid cell if we are not at the edge of the domain
        if curr_state_loc[1] != successor_state_loc[1] - 1 and curr_state_loc[1] != self.size[1] - 1:
            return 0.0
            
        # Periodic boundaries for East-West direction... we loop back around if we go off the Eastern edge
        if curr_state_loc[1] == self.size[1] - 1 and successor_state_loc[1] != 0:
            return 0.0

        # STAY and IMAGE cannot shift focus North-South
        if (action == 'STAY' or action == 'IMAGE') and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # At the bottom, SOUTH does nothing
        if action == 'SOUTH' and (curr_state_loc[0] == self.size[0] - 1) and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # Otherwise, it always goes south by one cell
        if action == 'SOUTH' and (curr_state_loc[0] != self.size[0] - 1) and (curr_state_loc[0] != successor_state_loc[0] - 1):
            return 0.0

        # At the top, NORTH does nothing
        if action == 'NORTH' and (curr_state_loc[0] == 0) and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # Otherwise, it always goes north by one cell
        if action == 'NORTH' and (curr_state_loc[0] != 0) and (curr_state_loc[0] != successor_state_loc[0] + 1):
            return 0.0

        weather_transition_prob = 1.0
        for loc in curr_state_weather:
            current_loc_weather = curr_state_weather[loc]
            new_loc_weather = successor_state_weather[loc]

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

        return weather_transition_prob

    def reward_function(self, state, action):
        curr_state_loc, curr_state_weather = self.__stateFactorsFromInt(state)
        if curr_state_loc in curr_state_weather and ACTION_DETAILS[action] == 'IMAGE':
            return 1.0 + 1.0 * curr_state_weather[curr_state_loc]
        elif curr_state_loc not in curr_state_weather and ACTION_DETAILS[action] == 'IMAGE':
            return -0.1
        return -0.01

    def start_state_function(self, state):
        return 1.0 / len(self.states()) 
