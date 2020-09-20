import policy_sketch_refine
import printer
import utils
from grid_world_abstract_mdp import GridWorldAbstractMDP
from grid_world_mdp import GridWorldMDP

GRID_WORLD_WIDTH = 20
GRID_WORLD_HEIGHT = 20
WALL_PROBABILITY = 0.2

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_STATE = 0

GAMMA = 0.99
RELAX_INFEASIBLE = False


# TODO: Clean this simulator up some more
def main():
    print("========== Initialization ================================")

    print("Setting up the grid world...")
    grid_world = utils.generate_random_grid_world(GRID_WORLD_WIDTH, GRID_WORLD_HEIGHT, WALL_PROBABILITY)

    print("Setting up the grid world MDP...")
    ground_mdp = GridWorldMDP(grid_world)

    print("Setting up the grid world abstract MDP...")
    abstract_mdp = GridWorldAbstractMDP(ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)

    print("Setting up the initial state, abstract state, and action...")
    current_state = INITIAL_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    current_action = None

    print("Setting up the initial policy...")
    solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
    values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

    print("Setting up visualization information...")
    visited_states = []
    expanded_states = {}

    while current_action != 'STAY':
        print("========== Simulator =====================================")

        print("Current State:", current_state)
        print("Current Abstract State:", current_abstract_state)

        ground_states = abstract_mdp.get_ground_states([current_abstract_state])

        for ground_state in ground_states:
            expanded_states[ground_state] = policy[ground_state]

        if current_state not in ground_states:
            current_abstract_state = abstract_mdp.get_abstract_state(current_state)
            print("New Abstract State:", current_abstract_state)

            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
            values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
            policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

        visited_states.append(current_state)

        current_action = policy[current_state]
        print("Current Action:", current_action)

        printer.print_grid_world_policy(grid_world, policy, visited_states=visited_states, expanded_states=expanded_states)

        current_state = utils.get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
