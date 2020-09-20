import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = None
POINTS_OF_INTEREST = None
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_STATE = None

GAMMA = 0.99
RELAX_INFEASIBLE = False


# TODO: Clean this simulator up some more
def main():
    print("========== Initialization ================================")

    print("Setting up the earth observation MDP...")
    ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)

    printer.print_earth_observation_domain(ground_mdp, 5)

    # print("Setting up the abstract earth observation MDP...")
    # abstract_mdp = EarthObservationAbstractMDP(ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)

    # print("Setting up the initial state, abstract state, and action...")
    # current_state = INITIAL_STATE
    # current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    # current_action = None

    # print("Setting up the initial policy...")
    # solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
    # values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    # policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

    # while True:
    #     print("========== Simulator =====================================")

    #     print("Current State:", current_state)
    #     print("Current Abstract State:", current_abstract_state)

    #     if current_state not in abstract_mdp.get_ground_states([current_abstract_state]):
    #         current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    #         print("New Abstract State:", current_abstract_state)

    #         solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
    #         values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    #         policy = utils.get_ground_policy(values, ground_mdp, GAMMA)

    #     current_action = policy[current_state]
    #     print("Current Action:", current_action)

    #     printer.print_grid_world_policy(grid_world, policy, current_state)

    #     current_state = utils.get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
