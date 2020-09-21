import logging
import time

import policy_sketch_refine
import printer
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = (4, 4)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 2
ABSTRACT_STATE_HEIGHT = 2

INITIAL_STATE = 0

GAMMA = 0.99
RELAX_INFEASIBLE = False

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


# TODO: Clean this simulator up some more
# TODO: Cache all results so that we only ever do this for unique combinations of abstract/ground states
def main():
    start = time.time()
    ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(ground_mdp.states()), len(ground_mdp.actions()), time.time() - start)

    start = time.time()
    abstract_mdp = EarthObservationAbstractMDP(ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(abstract_mdp.states()), len(abstract_mdp.actions()), time.time() - start)

    current_state = INITIAL_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_state)
    current_action = None
    logging.info("Initialized the current state: [%s]", current_state)
    logging.info("Initialized the current abstract state: [%s]", current_abstract_state)
    logging.info("Initialized the current action: [%s]", current_action)

    visited_states = []

    logging.info("Activating the simulator...")
    while True:
        if current_action is None or current_state not in abstract_mdp.get_ground_states([current_abstract_state]):
            current_abstract_state = abstract_mdp.get_abstract_state(current_state)
            logging.info("Encountered a new abstract state: [%s]", current_abstract_state)

            logging.info("Starting the policy sketch refine algorithm...")
            start = time.time()
            solution = policy_sketch_refine.solve(ground_mdp, abstract_mdp, current_abstract_state, GAMMA, RELAX_INFEASIBLE)
            logging.info("Finished the policy sketch refine algorithm: [time=%f]", time.time() - start)

            start = time.time()
            values = utils.get_ground_entities(solution['values'], ground_mdp, abstract_mdp)
            logging.info("Calculated the values from the solution of policy sketch refine: [time=%f]", time.time() - start)

            start = time.time()
            policy = utils.get_ground_policy(values, ground_mdp, GAMMA)
            logging.info("Calculated the policy from the values: [time=%f]", time.time() - start)

        visited_states.append(current_state)

        expanded_state_policy = {}
        for ground_state in abstract_mdp.get_ground_states([current_abstract_state]):
            expanded_state_policy[ground_state] = policy[ground_state]

        current_action = policy[current_state]

        logging.info("Current State: [%s]", current_state)
        logging.info("Current Abstract State: [%s]", current_abstract_state)
        logging.info("Current Action: [%s]", current_action)

        printer.print_earth_observation_policy(ground_mdp, policy, visited_states=visited_states, expanded_state_policy=expanded_state_policy)

        current_state = utils.get_successor_state(current_state, current_action, ground_mdp)


if __name__ == '__main__':
    main()
