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


def get_ground_policy(abstract_policy, mdp, abstract_mdp):
    ground_policy = {}

    for ground_state in mdp.states():
        abstract_state = abstract_mdp.get_abstract_state(ground_state)
        ground_policy[ground_state] = abstract_policy[abstract_state]

    return ground_policy
