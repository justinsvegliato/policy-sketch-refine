import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP
import time
import cplex_mdp_solver

SIZE = (6, 6)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_STATE = 0

GAMMA = 0.99
RELAX_INFEASIBLE = False


# TODO: Clean this simulator up some more
def main():
    print("========== Initialization ================================")

    print("Setting up the earth observation MDP...")
    ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)

    # print(ground_mdp.poi_description)
    # printer.print_reward_function(ground_mdp)
    # print("Calculating the policy for the ground MDP...")
    # solution = cplex_mdp_solver.solve(ground_mdp, 0.99, {}, False)
    # printer.print_earth_observation_policy(ground_mdp, solution['policy'], visited_states=[257], expanded_state_policy={})

    print("Setting up the abstract earth observation MDP...")
    abstract_mdp = EarthObservationAbstractMDP(ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)

    print("Setting up the initial state, abstract state, and action...")
    current_state = INITIAL_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    current_action = None

    print("Calculating the policy...")
    start_time = time.time()
    solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
    # values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
    # policy = utils.get_ground_policy(values, ground_mdp, GAMMA)
    policy = utils.get_ground_entities(solution['policy'], ground_mdp, abstract_mdp)
    print(f"Calculated the policy in {time.time() - start_time} seconds")

    print("Setting up the visualization information...")
    visited_states = []

    while True:
        print("========== Simulator =====================================")

        print("Current State:", current_state)
        print("Current Abstract State:", current_abstract_state)

        if current_state not in abstract_mdp.get_ground_states([current_abstract_state]):
            current_abstract_state = abstract_mdp.get_abstract_state(current_state)
            print("New Abstract State:", current_abstract_state)

            print("Calculating the policy...")
            start_time = time.time()
            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
            # values = utils.get_ground_values(solution['values'], ground_mdp, abstract_mdp)
            # policy = utils.get_ground_policy(values, ground_mdp, GAMMA)
            policy = utils.get_ground_entities(solution['policy'], ground_mdp, abstract_mdp)
            print(f"Calculated the policy in {time.time() - start_time} seconds")

        visited_states.append(current_state)

        expanded_state_policy = {}
        for ground_state in abstract_mdp.get_ground_states([current_abstract_state]):
            expanded_state_policy[ground_state] = policy[ground_state]

        current_action = policy[current_state]

        print("Current Action:", current_action)

        printer.print_earth_observation_policy(ground_mdp, policy, visited_states=visited_states, expanded_state_policy=expanded_state_policy)

        current_state = utils.get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
