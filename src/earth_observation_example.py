import logging
import time

import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = (3, 6)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_STATE = 0

GAMMA = 0.99
RELAX_INFEASIBLE = False

SLEEP_DURATION = 1.0

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def main():
    start = time.time()
    ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(ground_mdp.states()), len(ground_mdp.actions()), time.time() - start)

    start = time.time()
    abstract_mdp = EarthObservationAbstractMDP(ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(abstract_mdp.states()), len(abstract_mdp.actions()), time.time() - start)

    printer.print_transition_function(abstract_mdp)

    current_state = INITIAL_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    logging.info("Initialized the current state: [%s]", current_state)
    logging.info("Initialized the current abstract state: [%s]", current_abstract_state)

    state_history = []
    policy_cache = {}

    logging.info("Activating the simulator...")
    while True:
        ground_states = abstract_mdp.get_ground_states([current_abstract_state])

        if current_state not in policy_cache:
            logging.info("Encountered a new abstract state: [%s]", current_abstract_state)

            logging.info("Starting the policy sketch refine algorithm...")
            start = time.time()
            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
            logging.info("Finished the policy sketch refine algorithm: [time=%f]", time.time() - start)

            start = time.time()
            values = utils.get_ground_entities(solution['values'], ground_mdp, abstract_mdp)
            logging.info("Calculated the values from the solution of policy sketch refine: [time=%f]", time.time() - start)

            start = time.time()
            policy = utils.get_ground_policy(values, ground_mdp, abstract_mdp, ground_states, current_abstract_state, GAMMA)
            logging.info("Calculated the policy from the values: [time=%f]", time.time() - start)

            logging.info("Cached the ground states for the new abstract state: [%s]", current_abstract_state)
            for ground_state in ground_states:
                policy_cache[ground_state] = policy[ground_state]

        state_history.append(current_state)

        expanded_state_policy = {}
        for ground_state in ground_states:
            expanded_state_policy[ground_state] = policy_cache[ground_state]

        current_action = policy_cache[current_state]

        logging.info("Current State: [%s]", current_state)
        logging.info("Current Abstract State: [%s]", current_abstract_state)
        logging.info("Current Action: [%s]", current_action)

        printer.print_earth_observation_policy(ground_mdp, state_history=state_history, expanded_state_policy=expanded_state_policy, policy_cache=policy_cache)

        current_state = utils.get_successor_state(current_state, current_action, ground_mdp)
        current_abstract_state = abstract_mdp.get_abstract_state(current_state)

        time.sleep(SLEEP_DURATION)


if __name__ == '__main__':
    main()
