import cplex_mdp_solver
from abstract_mdp import AbstractMDP
from partially_abstract_mdp import PartiallyAbstractMDP


def solve(ground_mdp, gamma):
    abstract_mdp = AbstractMDP(ground_mdp, 0.9, 'MEAN')

    abstract_solution = cplex_mdp_solver.solve(abstract_mdp, gamma)

    ground_values = {}

    for abstract_state in abstract_mdp.states():
        constant_values = {}
        for constant_abstract_state in abstract_mdp.states():
            if constant_abstract_state != abstract_state:
                constant_values[abstract_state] = abstract_solution['values'][constant_abstract_state]

        partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, abstract_state)
        partially_abstract_solution = cplex_mdp_solver.solve(partially_abstract_mdp, 0.99, constant_values)

        for ground_state in ground_mdp.states():
            if ground_state in partially_abstract_solution['values']:
                ground_values[ground_state] = partially_abstract_solution['values'][ground_state]

    return ground_values
