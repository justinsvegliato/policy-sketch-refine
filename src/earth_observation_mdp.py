import copy
import math
from random import randint

ACTIONS = ['STAY', 'NORTH', 'SOUTH', 'IMAGE']

WEATHER_GETS_WORSE_PROBABILITY = 0.1
WEATHER_GETS_BETTER_PROBABILITY = 0.1
WEATHER_STAYS_SAME_PROBABILITY = 0.8

DEFAULT_SIZE = (6, 6)
DEFAULT_NUM_POI = 2

MIN_VISIBILITY = 0
MAX_VISIBILITY = 2

VISIBILITY_FIDELITY = MAX_VISIBILITY - MIN_VISIBILITY + 1


class EarthObservationMDP:
    def __init__(self, size=DEFAULT_SIZE, points_of_interest=None, visibility=None):
        # Create a dictionary ({(x, y): vis, ...}) containing the location tuple and starting visibility for each POI
        self.point_of_interest_description = {}

        # Set the number of rows and columns to the parameter or the default size
        self.num_rows = size[0]
        self.num_cols = size[1]

        # Set the points of interest in one of three different ways
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
            print("Failed to parse the point of interest argument")

        # Set the visibility in one of three different wayss
        if visibility is None:
            self.__init_random_visibility()
        elif isinstance(visibility, int):
            self.__init_constant_visibility(visibility)
        elif isinstance(visibility, dict):
            self.__init_exact_visibility(visibility)
        else:
            print("Failed to parse the visibility argument")

    def __init_random_points_of_interest(self):
        while len(self.point_of_interest_description) < self.num_points_of_interest:
            random_row = randint(0, self.num_rows - 1)
            random_col = randint(0, self.num_cols - 1)
            random_location = (random_row, random_col)
            self.point_of_interest_description[random_location] = 0

    def __init_set_points_of_interest(self, points_of_interest):
        for point in points_of_interest:
            self.point_of_interest_description[point] = 0

    def __init_random_visibility(self):
        for point in self.point_of_interest_description:
            random_visibility = randint(MIN_VISIBILITY, MAX_VISIBILITY)
            self.point_of_interest_description[point] = random_visibility

    def __init_constant_visibility(self, visibility):
        visibility = max(MIN_VISIBILITY, visibility)
        visibility = min(MAX_VISIBILITY, visibility)
        for point in self.point_of_interest_description:
            self.point_of_interest_description[point] = visibility

    def __init_exact_visibility(self, visibility):
        self.point_of_interest_description = visibility

    # TODO: Simplify/clean this function - Samer is too smart for me
    def get_state_factors_from_state(self, state):
        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest

        # Calculate the index of the location
        num_weather_statuses = pow(VISIBILITY_FIDELITY, power)
        location_id = math.floor(state / num_weather_statuses)
        row = math.floor(location_id / self.num_cols)
        col = location_id - row * self.num_cols
        # col = location_id % self.num_rows
        location = (row, col)

        # Calculate the index of the weather status
        weather_status = copy.deepcopy(self.point_of_interest_description)

        assert (power == len(weather_status)), "Inconsistent number of points of interest"

        locations = sorted(list(weather_status.keys()))

        # Overwrite values with whatever the state is representing
        weather_id = state % pow(base, power)
        for i in range(power - 1, -1, -1):
            location_weather = math.floor(weather_id / pow(base, i))
            weather_status[locations[i]] = location_weather
            weather_id = weather_id % pow(base, i)

        return location, weather_status

    # TODO: Simplify/clean this function - Samer is too smart for me
    def get_state_from_state_factors(self, location, poi_weather):
        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        weather_expansion_factor = pow(base, power)
        
        location_id = weather_expansion_factor * (location[0] * self.num_cols + location[1])

        locations = sorted(list(poi_weather.keys()))
        assert (power == len(poi_weather)), "Inconsistent number of points of interest"

        weather_id = 0
        for i in range(power - 1, -1, -1):
            weather_id += poi_weather[locations[i]] * pow(base, i)

        state = location_id + weather_id

        return state

    def get_num_point_of_interests(self):
        return self.num_points_of_interest

    def get_visibility_fidelity(self):
        return VISIBILITY_FIDELITY

    def get_points_of_interest(self):
        return list(self.point_of_interest_description.keys())

    def width(self):
        return self.num_cols

    def height(self):
        return self.num_rows

    def states(self):
        nums_locations = self.num_rows * self.num_cols

        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        num_weather_statuses = pow(base, power)

        num_states = nums_locations * num_weather_statuses

        return list(range(num_states))

    def actions(self):
        return ACTIONS

    def transition_function(self, state, action, successor_state):
        location, weather_status = self.get_state_factors_from_state(state)
        successor_location, successor_weather_status = self.get_state_factors_from_state(successor_state)

        # Move east by one grid cell if we are not at the edge of the domain
        if location[1] != successor_location[1] - 1 and location[1] != self.num_cols - 1:
            return 0.0

        # Loop back around if we go off the eastern edge (i.e., periodic boundaries for the east to west direction)
        if location[1] == self.num_cols - 1 and successor_location[1] != 0:
            return 0.0

        # STAY and IMAGE cannot shift focus north-south
        if (action == 'STAY' or action == 'IMAGE') and (location[0] != successor_location[0]):
            return 0.0

        # SOUTH does nothing at the bottom
        if action == 'SOUTH' and (location[0] == self.num_rows - 1) and (location[0] != successor_location[0]):
            return 0.0

        # SOUTH always goes south by one cell otherwise
        if action == 'SOUTH' and (location[0] != self.num_rows - 1) and (location[0] != successor_location[0] - 1):
            return 0.0

        # NORTH does nothing at the top
        if action == 'NORTH' and (location[0] == 0) and (location[0] != successor_location[0]):
            return 0.0

        # NORTH always goes north by one cell otherwise
        if action == 'NORTH' and (location[0] != 0) and (location[0] != successor_location[0] + 1):
            return 0.0

        weather_transition_probability = 1.0
        for location in weather_status:
            location_weather = weather_status[location]
            successor_location_weather = successor_weather_status[location]

            assert (location_weather >= MIN_VISIBILITY and location_weather <= MAX_VISIBILITY), "Bad weather value in current state"
            assert (successor_location_weather >= MIN_VISIBILITY and successor_location_weather <= MAX_VISIBILITY), "Bad weather value in successor state"

            # Weather in our model cannot change from good to bad immediately
            if abs(successor_location_weather - location_weather) > 1:
                return 0

            if location_weather == MIN_VISIBILITY:
                # Weather cannot get worse than minimum visibility
                if successor_location_weather == location_weather:
                    weather_transition_probability *= WEATHER_GETS_WORSE_PROBABILITY + WEATHER_STAYS_SAME_PROBABILITY
                else:
                    weather_transition_probability *= WEATHER_GETS_BETTER_PROBABILITY

            elif location_weather == MAX_VISIBILITY:
                # Weather cannot get better than maximum visibility
                if successor_location_weather == location_weather:
                    weather_transition_probability *= WEATHER_GETS_BETTER_PROBABILITY + WEATHER_STAYS_SAME_PROBABILITY
                else:
                    weather_transition_probability *= WEATHER_GETS_WORSE_PROBABILITY

            else:
                if successor_location_weather == location_weather:
                    weather_transition_probability *= WEATHER_STAYS_SAME_PROBABILITY
                elif successor_location_weather > location_weather:
                    weather_transition_probability *= WEATHER_GETS_BETTER_PROBABILITY
                else:
                    weather_transition_probability *= WEATHER_GETS_WORSE_PROBABILITY

        return weather_transition_probability

    # TODO: Determine the correct reward function
    def reward_function(self, state, action):
        location, weather_status = self.get_state_factors_from_state(state)

        if location in weather_status and action == 'IMAGE':
            return 1.0 + 3.0 * weather_status[location]

        if location not in weather_status and action == 'IMAGE':
            return -0.1

        return 0

    def start_state_function(self, _):
        return 1.0 / len(self.states())
