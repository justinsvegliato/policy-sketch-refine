import cplex_mdp_solver
import utils
from partially_abstract_mdp import PartiallyAbstractMDP

GAMMA = 0.99
RELAX_INFEASIBLE = False


def sketch(abstract_mdp):
    return cplex_mdp_solver.solve(abstract_mdp, GAMMA, relax_infeasible=RELAX_INFEASIBLE)


def refine(ground_mdp, abstract_mdp, abstract_state, sketched_solution):
    refined_solution = None

    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, [abstract_state])

    abstract_state_set = set(abstract_mdp.states())
    constant_abstract_state_set = abstract_state_set - {abstract_state}
    variable_abstract_state_set = abstract_state_set - constant_abstract_state_set

    print('Abstact State:', abstract_state)
    print('Abstract State Set:', abstract_state_set)
    print('Constant Abstract State Set:', constant_abstract_state_set)
    print('Variable Abstract State Set:', variable_abstract_state_set)
    print('=======')

    while True:
        constant_state_values = {}
        for partially_abstract_state in partially_abstract_mdp.states():
            if partially_abstract_state in constant_abstract_state_set:
                constant_state_values[partially_abstract_state] = sketched_solution['values'][partially_abstract_state]

        refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, GAMMA, constant_state_values=constant_state_values, relax_infeasible=RELAX_INFEASIBLE)

        if refined_solution:
            for constant_abstract_state in constant_abstract_state_set:
                refined_solution['values'][constant_abstract_state] = sketched_solution['values'][constant_abstract_state]
                refined_solution['policy'][constant_abstract_state] = sketched_solution['policy'][constant_abstract_state]
            break

        if not constant_abstract_state_set:
            break

        successor_abstract_state_set = utils.get_successor_state_set(abstract_mdp, variable_abstract_state_set)
        constant_abstract_state_set -= successor_abstract_state_set
        variable_abstract_state_set = abstract_state_set - constant_abstract_state_set

        print('Successor Abstract State Set:', successor_abstract_state_set)
        print('Constant Abstract State Set:', constant_abstract_state_set)
        print('Variable Abstract State Set:', variable_abstract_state_set)
        print('=======')

    return refined_solution


def solve(ground_mdp, abstract_mdp, abstract_state):
    sketched_solution = sketch(abstract_mdp)
    return refine(ground_mdp, abstract_mdp, abstract_state, sketched_solution)
