import math
from random import randint

ACTIONS = ['STAY', 'NORTH', 'SOUTH', 'IMAGE']

WEATHER_GETS_WORSE_PROBABILITY = 0.1
WEATHER_GETS_BETTER_PROBABILITY = 0.1
WEATHER_STAYS_SAME_PROBABILITY = 0.8

DEFAULT_SIZE = (6, 6)
DEFAULT_NUM_POI = 2

MIN_VISIBILITY = 0
MAX_VISIBILITY = 3

VISIBILITY_FIDELITY = MAX_VISIBILITY - MIN_VISIBILITY + 1


class EarthObservationMDP:
    def __init__(self, size=DEFAULT_SIZE, points_of_interest=None, visibility=None):
        # Create a dictionary ({(x, y): vis, ...}) containing the location tuple and starting visibility for each POI
        self.point_of_interest_description = {}

        # Set the number of rows and columns to the parameter or the default size
        self.num_rows = size[0]
        self.num_cols = size[1]

        self.state_registry = {}

        self.state_space = None

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
            self.__init_specified_points_of_interest(points_of_interest)
        else:
            assert "Failed to parse the point of interest argument"

        # Set the visibility in one of three different wayss
        if visibility is None:
            self.__init_random_visibility()
        elif isinstance(visibility, int):
            self.__init_constant_visibility(visibility)
        elif isinstance(visibility, dict):
            self.__init_exact_visibility(visibility)
        else:
            assert "Failed to parse the visibility argument"

    def __init_random_points_of_interest(self):
        while len(self.point_of_interest_description) < self.num_points_of_interest:
            random_row = randint(0, self.num_rows - 1)
            random_col = randint(0, self.num_cols - 1)
            random_location = (random_row, random_col)
            self.point_of_interest_description[random_location] = 0

    def __init_specified_points_of_interest(self, points_of_interest):
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

    def get_state_factors_from_state(self, state):
        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        num_weather_statuses = pow(base, power)

        # Calculate the index of the location
        location_id = math.floor(state / num_weather_statuses)
        row = math.floor(location_id / self.num_cols)
        col = location_id - row * self.num_cols
        location = (row, col)

        weather_status = {}

        # Sort the locations array to enforce an ordering
        locations = sorted(self.point_of_interest_description.keys())

        # Overwrite values with whatever the state is representing
        weather_id = state % num_weather_statuses
        for i in range(power - 1, -1, -1):
            base_to_the_i = pow(base, i)
            location_weather = math.floor(weather_id / base_to_the_i)
            weather_status[locations[i]] = location_weather
            weather_id = weather_id % base_to_the_i

        return location, weather_status

    def get_state_from_state_factors(self, location, weather_status):
        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        num_weather_statuses = pow(base, power)

        assert (len(weather_status) == power), "Inconsistent number of points of interest"

        # Calculate the starting index of the location
        location_id = num_weather_statuses * (location[0] * self.num_cols + location[1])

        # Sort the locations array to enforce an ordering
        locations = sorted(list(weather_status.keys()))

        # Calculate the offset index of the weather
        weather_id = 0
        for i in range(power - 1, -1, -1):
            weather_id += weather_status[locations[i]] * pow(base, i)

        # Add the offset index of the weather to the starting index of the location
        return location_id + weather_id

    def get_successors(self, state, action):
        successors = []

        # TODO: do the weather part for even more speedup / accuracy
        location, weather_status = self.get_state_factors_from_state(state)

        successor_location = (0, 0)

        # Northern-most row
        if location[0] == 0:
            if location[1] == self.num_cols - 1:
                successor_location = (location[0], 0) if action in ('NORTH', 'STAY', 'IMAGE') else (location[0] + 1, 0)
            else:
                successor_location = (location[0], location[1] + 1) if action in ('NORTH', 'STAY', 'IMAGE') else (location[0] + 1, location[1] + 1)

        # Southern-most row
        elif location[0] == self.num_rows - 1:
            if location[1] == self.num_cols - 1:
                successor_location = (location[0], 0) if action in ('SOUTH', 'STAY', 'IMAGE') else (location[0] - 1, 0)
            else:
                successor_location = (location[0], location[1] + 1) if action in ('SOUTH', 'STAY', 'IMAGE') else (location[0] - 1, location[1] + 1)

        # Any interior row
        else:
            if location[1] == self.num_cols - 1:
                succs = {
                    'NORTH': (location[0] - 1, 0), 
                    'SOUTH': (location[0] + 1, 0), 
                    'STAY': (location[0], 0), 
                    'IMAGE': (location[0], 0) }
                successor_location = succs[action]
            else:
                succs = {
                    'NORTH': (location[0] - 1, location[1] + 1), 
                    'SOUTH': (location[0] + 1, location[1] + 1), 
                    'STAY': (location[0], location[1] + 1), 
                    'IMAGE': (location[0], location[1] + 1) }
                successor_location = succs[action]
        
        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        num_weather_statuses = pow(base, power)

        successor_location_id = num_weather_statuses * (successor_location[0] * self.num_cols + successor_location[1])
        successors = range(successor_location_id, successor_location_id + num_weather_statuses)

        return set(successors)

    def get_num_point_of_interests(self):
        return self.num_points_of_interest

    def get_visibility_fidelity(self):
        return VISIBILITY_FIDELITY

    def width(self):
        return self.num_cols

    def height(self):
        return self.num_rows

    def states(self):
        if self.state_space:
            return self.state_space

        nums_locations = self.num_rows * self.num_cols

        base = VISIBILITY_FIDELITY
        power = self.num_points_of_interest
        num_weather_statuses = pow(base, power)

        num_states = nums_locations * num_weather_statuses

        self.state_space = list(range(num_states))

        return self.state_space

    def actions(self):
        return ACTIONS

    def transition_function(self, state, action, successor_state):
        if state not in self.state_registry:
            self.state_registry[state] = self.get_state_factors_from_state(state)

        if successor_state not in self.state_registry:
            self.state_registry[successor_state] = self.get_state_factors_from_state(successor_state)

        location, weather_status = self.state_registry[state]
        successor_location, successor_weather_status = self.state_registry[successor_state]

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
