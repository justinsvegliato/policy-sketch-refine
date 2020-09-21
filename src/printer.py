import sys

from termcolor import colored


def print_states(mdp):
    print("States:")

    for index, state in enumerate(mdp.states()):
        print(f"  State {index}: {state}")


def print_actions(mdp):
    print("Actions:")

    for index, action in enumerate(mdp.actions()):
        print(f"  Action {index}: {action}")


def print_transition_function(mdp):
    print("Transition Function:")

    is_valid = True

    for state in mdp.states():
        for action in mdp.actions():
            print(f"  Transition: ({state}, {action})")

            total_probability = 0

            for successor_state in mdp.states():
                probability = mdp.transition_function(
                    state, action, successor_state)

                total_probability += probability

                if probability > 0:
                    print(
                        f"    Successor State: {successor_state} -> {probability}")

            is_valid = is_valid and 0.99 <= total_probability <= 1.01
            print(f"    Total Probability: {total_probability}")

            if not is_valid:
                return

    print(f"  Is Valid: {is_valid}")


def print_reward_function(mdp):
    print("Reward Function:")

    for state in mdp.states():
        print(f"  State: {mdp.state_factors_from_int(state)}")

        for action in mdp.actions():
            reward = mdp.reward_function(state, action)
            print(f"    Action: {action} -> {reward}")


def print_start_state_function(mdp):
    print("Start State Function:")

    total_probability = 0

    for state in mdp.states():
        probability = mdp.start_state_function(state)
        total_probability += probability
        print(f"  State {state}: {probability}")

    print(f"  Total Probability: {total_probability}")

    is_valid = total_probability == 1.0
    print(f"  Is Valid: {is_valid}")


def print_mdp(mdp):
    print_states(mdp)
    print_actions(mdp)
    print_transition_function(mdp)
    print_reward_function(mdp)
    print_start_state_function(mdp)


def print_grid_world_policy(grid_world, policy, visited_states=[], expanded_states={}):
    SYMBOLS = {
        'STAY': '\u2205',
        'NORTH': '\u2191',
        'EAST': '\u2192',
        'SOUTH': '\u2193',
        'WEST': '\u2190'
    }

    height = len(grid_world)
    width = len(grid_world[0])

    for row in range(height):
        text = ""

        for column in range(width):
            state = len(grid_world[row]) * row + column

            symbol = None
            if grid_world[row][column] == 'W':
                symbol = '\u25A0'
            else:
                symbol = SYMBOLS[policy[state] if state not in expanded_states else expanded_states[state]]

            if state in visited_states:
                symbol = colored(symbol, 'red')
            elif state in expanded_states:
                symbol = colored(symbol, 'blue')

            text += symbol
            text += "  "

        print(f"{text}")


def print_grid_world_values(grid_world, values):
    height = len(grid_world)
    width = len(grid_world[0])

    for row in range(height):
        text = ""

        for column in range(width):
            state = len(grid_world[row]) * row + column
            if grid_world[row][column] == 'W':
                text += "{:^5s}".format('\u25A0')
            else:
                text += "{:^5.2f}".format(values[state])
            text += "  "

        print(f"{text}")


def print_earth_observation_policy(earth_observation_mdp, state_history=[], expanded_state_policy={}, policy_cache={}):
    BORDER_SIZE = 150
    SYMBOLS = {
        0: '\u00b7',
        1: '\u205a',
        2: '\u22ee',
        'STAY': '\u2205',
        'NORTH': '\u2191',
        'SOUTH': '\u2193',
        'IMAGE': '\u25A1'
    }

    print("=" * BORDER_SIZE)

    height = earth_observation_mdp.height()
    width = earth_observation_mdp.width()

    for row in range(height):
        text = ""

        for column in range(width):
            location = (row, column)

            current_state = state_history[-1]
            _, current_poi_weather = earth_observation_mdp.get_state_factors_from_state(current_state)

            state = earth_observation_mdp.get_state_from_state_factors(location, current_poi_weather)

            symbol = None
            if location in current_poi_weather:
                weather_symbol = SYMBOLS[current_poi_weather[location]]
                symbol = weather_symbol
            else:
                if state in policy_cache:
                    action = policy_cache[state]
                    symbol = SYMBOLS[action]
                else:
                    symbol = "\u2A09"

            if state == state_history[-1]:
                symbol = colored(symbol, 'red')
            elif state in expanded_state_policy:
                symbol = colored(symbol, 'blue')
            elif state in policy_cache:
                symbol = colored(symbol, 'green')

            text += symbol
            text += "  "

        print(f"{text}")

    print("=" * BORDER_SIZE)


def print_loading_bar(count, total, label):
    maximum_loading_bar_length = 60
    current_loading_bar_length = int(round(maximum_loading_bar_length * count / float(total)))

    percent = round(100.0 * count / float(total), 1)
    loading_bar = '#' * current_loading_bar_length + '-' * (maximum_loading_bar_length - current_loading_bar_length)

    sys.stdout.write('%s: [%s] %s%s %s\r' % (label, loading_bar, percent, '%', ''))
    sys.stdout.flush()
