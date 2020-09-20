import math


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
                probability = mdp.transition_function(state, action, successor_state)

                total_probability += probability

                if probability > 0:
                    print(f"    Successor State: {successor_state} -> {probability}")

            is_valid = is_valid and 0.99 <= total_probability <= 1.01
            print(f"    Total Probability: {total_probability}")

            if not is_valid:
                return

    print(f"  Is Valid: {is_valid}")


def print_reward_function(mdp):
    print("Reward Function:")

    for state in mdp.states():
        print(f"  State: {state}")

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


def print_grid_world_domain(grid_world, current_state):
    height = len(grid_world)
    width = len(grid_world[0])

    current_row = math.floor(current_state / width)
    current_column = current_state - current_row * width

    for row in range(height):
        text = ""

        for column in range(width):
            if row == current_row and column == current_column:
                text += "R"
            elif grid_world[row][column] == 'W':
                text += "\u25A0"
            elif grid_world[row][column] == 'G':
                text += "\u272A"
            elif grid_world[row][column] == 'S':
                text += "\u229B"
            else:
                text += "\u25A1"
            text += "  "

        print(f"{text}")


def print_grid_world_policy(grid_world, policy, current_state):
    symbols = {
        'STAY': '\u2205',
        'NORTH': '\u2191',
        'EAST': '\u2192',
        'SOUTH': '\u2193',
        'WEST': '\u2190'
    }

    height = len(grid_world)
    width = len(grid_world[0])

    current_row = math.floor(current_state / width)
    current_column = current_state - current_row * width

    for row in range(height):
        text = ""

        for column in range(width):
            state = len(grid_world[row]) * row + column
            if row == current_row and column == current_column:
                text += "R"
            elif grid_world[row][column] == 'W':
                text += "\u25A0"
            else:
                text += symbols[policy[state]]
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
                text += "{:^5s}".format("\u25A0")
            else:
                text += "{:^5.2f}".format(values[state])
            text += "  "

        print(f"{text}")
