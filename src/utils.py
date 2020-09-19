import random


def generate_random_grid_world(width, height, wall_probability):
    grid_world = [['O' for column in range(width)] for row in range(height)]

    for row in range(height):
        for column in range(width):
            probability = random.random()
            if probability < wall_probability:
                grid_world[row][column] = 'W'

    grid_world[height - 1][width - 1] = 'G'

    return grid_world


def get_ground_policy(abstract_policy, ground_mdp, abstract_mdp):
    ground_policy = {}

    for ground_state in ground_mdp.states():
        if ground_state in abstract_policy:
            ground_policy[ground_state] = abstract_policy[ground_state]
        else:
            abstract_state = abstract_mdp.get_abstract_state(ground_state)
            ground_policy[ground_state] = abstract_policy[abstract_state]

    return ground_policy


def get_successor_state_set(mdp, states):
    successor_state_set = set()

    for state in states:
        for action in mdp.actions():
            for successor_state in mdp.states():
                if mdp.transition_function(state, action, successor_state) > 0:
                    successor_state_set.add(successor_state)

    return successor_state_set
