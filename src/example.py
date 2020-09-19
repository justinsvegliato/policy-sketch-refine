import random

import policy_sketch_refine
import printer
import utils
from abstract_mdp import AbstractMDP
from grid_world_mdp import GridWorldMDP


def get_successor_state(current_state, current_action, mdp):
    probability_threshold = random.random()

    total_probability = 0

    for successor_state in mdp.states():
        transition_probability = mdp.transition_function(current_state, current_action, successor_state)

        total_probability += transition_probability

        if total_probability >= probability_threshold:
            return successor_state


def main():
    print("Setting up the grid world...")
    grid_world = utils.generate_random_grid_world(20, 20, 0.1)

    print("Setting up the ground MDP...")
    ground_mdp = GridWorldMDP(grid_world)

    print("Setting up the abstract MDP...")
    abstract_mdp = AbstractMDP(ground_mdp, 'MEAN', 3, 3)

    current_state = 0
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    current_action = None

    print('Current State:', current_state)
    print('Current Abstract State:', current_abstract_state)
    print('Current Action:', current_action)
    print("=======================================================")

    solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state)
    # policy = utils.get_ground_policy(solution['policy'], ground_mdp, abstract_mdp)
    values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    policy = utils.get_policy(values, ground_mdp, 0.99)

    printer.print_grid_world_domain(grid_world, current_state)

    while current_action != "STAY":
        print('Current State:', current_state)
        print('Current Abstract State:', current_abstract_state)

        if current_state not in abstract_mdp.get_ground_states([current_abstract_state]):
            current_abstract_state = abstract_mdp.get_abstract_state(current_state)
            print('New Abstract State:', current_abstract_state)

            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state)
            policy = utils.get_ground_policy(solution['policy'], ground_mdp, abstract_mdp)
            values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
            policy = utils.get_policy(values, ground_mdp, 0.99)

        current_action = policy[current_state]
        print('Current Action:', current_action)

        printer.print_grid_world_policy(grid_world, policy, current_state)
        print("=======================================================")

        current_state = get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
