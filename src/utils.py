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
