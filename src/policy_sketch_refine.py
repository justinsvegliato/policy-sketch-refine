import logging

import cplex_mdp_solver
import utils
from partially_abstract_mdp import PartiallyAbstractMDP

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-15s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def sketch(abstract_mdp, gamma, relax_infeasible):
    logging.info("Sketching the solution...")
    return cplex_mdp_solver.solve(abstract_mdp, gamma, constant_state_values={}, relax_infeasible=relax_infeasible)


def refine(ground_mdp, abstract_mdp, abstract_state, sketched_solution, gamma, relax_infeasible):
    logging.info("Refining the solution...")

    refined_solution = None

    logging.info("Building the PAMDP...")
    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, [abstract_state])
    logging.info("Built the PAMDP [states = %d, actions = %d]", len(partially_abstract_mdp.states()), len(partially_abstract_mdp.actions()))

    abstract_state_set = set(abstract_mdp.states())
    constant_abstract_state_set = abstract_state_set - {abstract_state}
    variable_abstract_state_set = abstract_state_set - constant_abstract_state_set
    successor_abstract_state_set = []

    while True:
        constant_state_values = {}
        for partially_abstract_state in partially_abstract_mdp.states():
            if partially_abstract_state in constant_abstract_state_set:
                constant_state_values[partially_abstract_state] = sketched_solution['values'][partially_abstract_state]

        logging.info("Running the CPLEX solver on the PAMDP...")
        refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma, constant_state_values=constant_state_values, relax_infeasible=relax_infeasible)
        logging.info("Finished running the CPLEX solver on the PAMDP")

        if refined_solution:
            logging.info("Found a solution to the PAMDP")
            for constant_abstract_state in constant_abstract_state_set:
                refined_solution['values'][constant_abstract_state] = sketched_solution['values'][constant_abstract_state]
                refined_solution['policy'][constant_abstract_state] = sketched_solution['policy'][constant_abstract_state]
            break

        if not constant_abstract_state_set:
            logging.error("Failed to find a solution to the PAMDP")
            break

        successor_abstract_state_set = utils.get_successor_state_set(abstract_mdp, variable_abstract_state_set)
        constant_abstract_state_set -= successor_abstract_state_set
        variable_abstract_state_set = abstract_state_set - constant_abstract_state_set

    return refined_solution


def solve(ground_mdp, abstract_mdp, abstract_state, gamma, relax_infeasible):
    sketched_solution = sketch(abstract_mdp, gamma, relax_infeasible)
    return refine(ground_mdp, abstract_mdp, abstract_state, sketched_solution, gamma, relax_infeasible)
