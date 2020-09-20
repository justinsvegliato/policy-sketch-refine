import copy
import math
from random import randint

ACTION_DETAILS = {
    'STAY': {},
    'NORTH': {},
    'SOUTH': {},
    'IMAGE': {},
}

PROB_WEATHER_GETS_WORSE = 0.1
PROB_WEATHER_GETS_BETTER = 0.1
PROB_WEATHER_STAYS_SAME = 0.8

#DEFAULT_SIZE = (12, 12)
DEFAULT_SIZE = (6, 6)

MIN_VISIBILITY = 0
MAX_VISIBILITY = 2

#DEFAULT_NUM_POI = 5
DEFAULT_NUM_POI = 2


class EarthObservationMDP:
    def __init__(self, size=None, points_of_interest=None, visibility=None):
        self.size = DEFAULT_SIZE if size is None else size

        # Create a dictionary ({(x, y): vis, ...}) containing the location tuple and starting visibility for each POI
        self.poi_description = {}

        # Set the points of interest
        self.num_points_of_interest = 0
        if points_of_interest is None:
            self.num_points_of_interest = DEFAULT_NUM_POI
            self.__init_random_points_of_interest()
        elif isinstance(points_of_interest, int):
            self.num_points_of_interest = points_of_interest
            self.__init_random_points_of_interest()
        elif isinstance(points_of_interest, list):
            self.num_points_of_interest = len(points_of_interest)
            self.__init_set_points_of_interest(points_of_interest)
        else:
            print("POI argument not parsed correctly")

        # Set the visibility
        if visibility is None:
            self.__init_random_visibility()
        elif isinstance(visibility, int):
            self.__init_constant_visibility(visibility)
        elif isinstance(visibility, dict):
            self.__init_exact_visibility(visibility)
        else:
            print("Visibility argument not parsed correctly")

        self.visibility_fidelity = MAX_VISIBILITY - MIN_VISIBILITY + 1

    def __init_random_points_of_interest(self):
        while len(self.poi_description) < self.num_points_of_interest:
            rand_lat = randint(0, self.size[0] - 1)
            rand_long = randint(0, self.size[1] - 1)
            rand_loc = (rand_long, rand_lat)
            self.poi_description[rand_loc] = 0

    def __init_set_points_of_interest(self, points_of_interest):
        for point in points_of_interest:
            self.poi_description[point] = 0

    def __init_random_visibility(self):
        for point in self.poi_description:
            rand_vis = randint(MIN_VISIBILITY, MAX_VISIBILITY)
            self.poi_description[point] = rand_vis

    def __init_constant_visibility(self, visibility):
        visibility = max(MIN_VISIBILITY, visibility)
        visibility = min(MAX_VISIBILITY, visibility)
        for point in self.poi_description:
            self.poi_description[point] = visibility

    def __init_exact_visibility(self, visibility):
        self.poi_description = visibility

    def state_factors_from_int(self, state_id):
        rows = self.size[0]
        cols = self.size[1]

        base = MAX_VISIBILITY - MIN_VISIBILITY + 1
        power = self.num_points_of_interest

        # Index location
        loc_id = math.floor(state_id / pow(base, power))
        latitude = math.floor(loc_id / cols)
        longitude = loc_id % rows
        location = (latitude, longitude)

        # Index weather
        poi_weather = copy.deepcopy(self.poi_description)

        locs = sorted(list(poi_weather.keys()))
        assert (power == len(poi_weather)), "Inconsistent number of points of interest"

        # Overwrite values with whatever the state_id is representing
        weather_id = state_id % pow(base, power)
        for i in range(power - 1, -1, -1):
            weather_at_loc = math.floor(weather_id / pow(base, i))
            poi_weather[locs[i]] = weather_at_loc
            weather_id = weather_id % pow(base, i)

        return location, poi_weather

    def int_from_state_factors(self, location, poi_weather):
        rows = self.size[0]
        cols = self.size[1]
        
        base = MAX_VISIBILITY - MIN_VISIBILITY + 1
        power = self.num_points_of_interest
        weather_expansion_factor = pow(base, power)
        
        location_id = weather_expansion_factor * (location[0] * cols + location[1])

        locs = sorted(list(poi_weather.keys()))
        assert (power == len(poi_weather)), "Inconsistent number of points of interest"

        weather_id = 0
        for i in range(power - 1, -1, -1):
            weather_id += poi_weather[locs[i]] * pow(base, i)

        state_id = location_id + weather_id

        return state_id

    def get_num_POI_num_vis(self):
        return self.num_points_of_interest, self.visibility_fidelity

    def get_POI_Locations(self):
        return list(self.poi_description.keys())

    def width(self):
        return self.size[1]

    def height(self):
        return self.size[0]

    def states(self):
        locations = self.size[0] * self.size[1]
        base = MAX_VISIBILITY - MIN_VISIBILITY + 1
        power = self.num_points_of_interest

        total_number_of_states = pow(base, power) * locations

        return list(range(total_number_of_states))

    def actions(self):
        return list(ACTION_DETAILS.keys())

    def transition_function(self, state, action, successor_state):
        curr_state_loc, curr_state_weather = self.state_factors_from_int(state)
        successor_state_loc, successor_state_weather = self.state_factors_from_int(successor_state)

        # Move East by one grid cell if we are not at the edge of the domain
        if curr_state_loc[1] != successor_state_loc[1] - 1 and curr_state_loc[1] != self.size[1] - 1:
            return 0.0

        # Loop back around if we go off the Eastern edge (i.e., periodic boundaries for the east to west direction)
        if curr_state_loc[1] == self.size[1] - 1 and successor_state_loc[1] != 0:
            return 0.0

        # STAY and IMAGE cannot shift focus North-South
        if (action == 'STAY' or action == 'IMAGE') and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # SOUTH does nothing at the bottom
        if action == 'SOUTH' and (curr_state_loc[0] == self.size[0] - 1) and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # SOUTH always goes south by one cell otherwise
        if action == 'SOUTH' and (curr_state_loc[0] != self.size[0] - 1) and (curr_state_loc[0] != successor_state_loc[0] - 1):
            return 0.0

        # NORTH does nothing at the top
        if action == 'NORTH' and (curr_state_loc[0] == 0) and (curr_state_loc[0] != successor_state_loc[0]):
            return 0.0

        # NORTH always goes north by one cell otherwise
        if action == 'NORTH' and (curr_state_loc[0] != 0) and (curr_state_loc[0] != successor_state_loc[0] + 1):
            return 0.0

        weather_transition_prob = 1.0
        for loc in curr_state_weather:
            current_loc_weather = curr_state_weather[loc]
            new_loc_weather = successor_state_weather[loc]

            assert (current_loc_weather >= MIN_VISIBILITY and current_loc_weather <= MAX_VISIBILITY), "Bad weather value in current state"
            assert (new_loc_weather >= MIN_VISIBILITY and new_loc_weather <= MAX_VISIBILITY), "Bad weather value in successor state"

            # Weather in our model cannot change from good to bad immediately
            if abs(new_loc_weather - current_loc_weather) > 1:
                return 0

            if current_loc_weather == MIN_VISIBILITY:
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
        curr_state_loc, curr_state_weather = self.state_factors_from_int(state)

        if curr_state_loc in curr_state_weather and action == 'IMAGE':
            return 1.0 + 1.0 * curr_state_weather[curr_state_loc]

        if curr_state_loc not in curr_state_weather and action == 'IMAGE':
            return -0.1

        return -0.01

    def start_state_function(self, _):
        return 1.0 / len(self.states())
