import policy_sketch_refine
import printer
import utils
from abstract_grid_world_mdp import AbstractGridWorldMDP
from grid_world_mdp import GridWorldMDP

GRID_WORLD_WIDTH = 20
GRID_WORLD_HEIGHT = 20
WALL_PROBABILITY = 0.1

ABSTRACTION = 'MEAN'
ABSTRACT_GRID_WORLD_WIDTH = 3
ABSTRACT_GRID_WORLD_HEIGHT = 3

INITIAL_STATE = 0

GAMMA = 0.99


def main():
    print("========== Initialization ================================")

    print("Setting up the grid world...")
    grid_world = utils.generate_random_grid_world(GRID_WORLD_WIDTH, GRID_WORLD_HEIGHT, WALL_PROBABILITY)

    print("Setting up the grid world MDP...")
    ground_mdp = GridWorldMDP(grid_world)

    print("Setting up the abstract grid world MDP...")
    abstract_mdp = AbstractGridWorldMDP(ground_mdp, ABSTRACTION, ABSTRACT_GRID_WORLD_WIDTH, ABSTRACT_GRID_WORLD_HEIGHT)

    print("Setting up the initial state, abstract state, and action...")
    current_state = INITIAL_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    current_action = None

    print("Setting up the initial policy...")
    solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state)
    values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

    # TODO: Clean this simulator some more
    while current_action != "STAY":
        print("========== Simulator =====================================")

        print('Current State:', current_state)
        print('Current Abstract State:', current_abstract_state)

        if current_state not in abstract_mdp.get_ground_states([current_abstract_state]):
            current_abstract_state = abstract_mdp.get_abstract_state(current_state)
            print('New Abstract State:', current_abstract_state)

            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state)
            values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
            policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

        current_action = policy[current_state]
        print('Current Action:', current_action)

        printer.print_grid_world_policy(grid_world, policy, current_state)

        current_state = utils.get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
