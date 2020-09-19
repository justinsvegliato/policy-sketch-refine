from pathlib import Path

import cplex
import numpy as np


class MemoryMDP:
    def __init__(self, mdp):
        self.states = mdp.states()
        self.actions = mdp.actions()

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        self.rewards = np.zeros(shape=(self.n_states, self.n_actions))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.rewards[state, action] = mdp.reward_function(self.states[state], self.actions[action])

        self.transition_probabilities = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                for successor_state in range(self.n_states):
                    self.transition_probabilities[state, action, successor_state] = mdp.transition_function(self.states[state], self.actions[action], self.states[successor_state])

        self.start_state_probabilities = np.zeros(self.n_states)
        for state in range(self.n_states):
            self.start_state_probabilities[state] = self.start_state_probabilities[state] = mdp.start_state_function(self.states[state])


def __validate(memory_mdp):
    assert memory_mdp.n_states is not None
    assert memory_mdp.n_actions is not None

    assert memory_mdp.states is not None
    assert memory_mdp.actions is not None
    assert memory_mdp.rewards is not None
    assert memory_mdp.transition_probabilities is not None
    assert memory_mdp.start_state_probabilities is not None

    assert memory_mdp.rewards.shape == (memory_mdp.n_states, memory_mdp.n_actions)
    assert memory_mdp.transition_probabilities.shape == (memory_mdp.n_states, memory_mdp.n_actions, memory_mdp.n_states)
    assert memory_mdp.start_state_probabilities.shape == (memory_mdp.n_states,)


def __set_variables(c, memory_mdp, constant_state_values):
    n_solving_states = memory_mdp.n_states - len(constant_state_values)
    print("CPLEX: adding {} variables".format(n_solving_states))
    # REMEMBER! Variable indices in CPLEX *always* start from 0!
    # TODO: Do we really need the upper and lower bounds?
    c.variables.add(types=[c.variables.type.continuous] * n_solving_states,
                    lb=[-10000] * n_solving_states,
                    ub=[10000] * n_solving_states)


def __set_objective(c, memory_mdp, constant_state_values):
    n_solving_states = memory_mdp.n_states - len(constant_state_values)

    solving_states_coefficients = []
    for i in range(memory_mdp.n_states):
        state = memory_mdp.states[i]
        if state not in constant_state_values:
            solving_states_coefficients.append(memory_mdp.start_state_probabilities[i])

    print("Constant states: {}".format(constant_state_values.keys()))
    print("Solving states coefficients:", solving_states_coefficients)
    print("N solving states: {}".format(n_solving_states))
    assert len(solving_states_coefficients) == n_solving_states

    c.objective.set_linear(enumerate(solving_states_coefficients))
    c.objective.set_sense(c.objective.sense.minimize)


def __set_constraints(program, memory_mdp, gamma, constant_state_values):
    lin_expressions = []
    right_hand_sides = []
    names = []

    n_solving_states = memory_mdp.n_states - len(constant_state_values)
    variables = range(n_solving_states)

    # There is one constraint for each (start_state, action) pair
    for i in range(memory_mdp.n_states):
        for j in range(memory_mdp.n_actions):
            # Here, we define 1 linear constraint for a (start_state, action) pair
            rhs = memory_mdp.rewards[i, j]

            start_state = memory_mdp.states[i]

            # If the start start state is "constant", discount its value from the r.h.s. of the constraint
            if start_state in constant_state_values:
                rhs -= constant_state_values[start_state]

            # Span all existing states as end_states:
            # - set the variables' coefficients corresponding to non-constant states
            # - or modify the r.h.s for constant states
            coefficients = []
            for k in range(memory_mdp.n_states):
                end_state = memory_mdp.states[k]

                # If end_state is "constant", use its value and transition to modify the r.h.s. of the constraint
                if end_state in constant_state_values:
                    rhs += gamma * memory_mdp.transition_probabilities[i, j, k] * constant_state_values[end_state]

                else:
                    # Set the end_state variable's coefficient

                    # If end_state is not start_state
                    if k != i:
                        coefficient = - gamma * memory_mdp.transition_probabilities[i, j, k]

                    # If end_state is start_state
                    else:
                        coefficient = 1 - gamma * memory_mdp.transition_probabilities[i, j, k]

                    coefficients.append(coefficient)

            if sum(coefficients) <= 0 < rhs:
                # FIXME: Why is this happening???
                continue

            # Avoid putting useless constraints (constraints with no variables)
            if all(c == 0 for c in coefficients):
                continue

            # Don't put bounds FIXME trying this for now to avoid inconsistent bound problems
            # if len([c for c in coefficients if c != 0]) == 1:
            #     continue

            # Append linear constraint
            # TODO: It may become necessary (for space complex.) to avoid giving CPLEX 0 coefficients altogether.
            lin_expressions.append([variables, coefficients])

            # The constraint's right-hand side is simply the reward
            right_hand_sides.append(float(rhs))

            names.append(f"{start_state}_{memory_mdp.actions[j]}")

    # Add all linear constraints to CPLEX at once
    program.linear_constraints.add(
        names=names,
        lin_expr=lin_expressions,
        rhs=right_hand_sides,
        senses=["G"] * len(lin_expressions))


def __get_policy(values, memory_mdp, gamma, constant_state_values=None):
    if constant_state_values is None:
        constant_state_values = {}

    policy = []

    n_solving_states = memory_mdp.n_states - len(constant_state_values)
    variables = range(n_solving_states)

    var_state_indices = []  # TODO: Get this as input "variable_states"?
    for i in range(memory_mdp.n_states):
        # We're not solving for the constant states
        if memory_mdp.states[i] not in constant_state_values:
            var_state_indices.append(i)

    assert len(variables) == len(values) == len(var_state_indices)

    for i in variables:
        best_action, best_action_value = None, None

        for j in range(memory_mdp.n_actions):
            action_value = memory_mdp.rewards[i, j] + gamma * np.sum(memory_mdp.transition_probabilities[i, j][var_state_indices] * values)
            if best_action_value is None or action_value > best_action_value:
                best_action = j
                best_action_value = action_value

        policy.append(best_action)

    return policy


def __create_problem(memory_mdp, gamma, constant_state_values):
    c = cplex.Cplex()

    __set_variables(c, memory_mdp, constant_state_values)
    __set_objective(c, memory_mdp, constant_state_values)
    __set_constraints(c, memory_mdp, gamma, constant_state_values)

    print("===== Program Details =============================================")
    print("{} variables".format(c.variables.get_num()))
    print("{} sense".format(c.objective.sense[c.objective.get_sense()]))
    print("{} linear coefficients".format(len(c.objective.get_linear())))
    print("{} linear constraints".format(c.linear_constraints.get_num()))
    print("Number of integer variables:", c.variables.get_num_integer())
    print("Variables upper bounds:", c.variables.get_upper_bounds())
    print("Variables lower bounds:", c.variables.get_lower_bounds())

    return c


def __solve_optimally(c):
    print("===== CPLEX Details ===============================================")
    print("===== SOLVE =======================================================")
    c.solve()
    print("===================================================================")

    success_statuses = [
        c.solution.status.MIP_optimal,
        c.solution.status.optimal_tolerance,
    ]

    infeasible_statuses = [
        c.solution.status.infeasible,
        c.solution.status.MIP_infeasible,
    ]

    status = c.solution.get_status()
    status_string = c.solution.get_status_string()
    print("CPLEX STATUS:", status, status_string)

    method = c.solution.get_method()
    print("CPLEX METHOD:", method, c.solution.method[method])

    if status in success_statuses:
        return "success"

    elif status in infeasible_statuses:
        return "infeasible"

    else:
        # TODO: Solution was not MIP optimal. It could be infeasible, suboptimal, etc. Check status.
        # Assuming problem was infeasible.
        return None


def __solve_feasibly(c):
    print("===== CPLEX Details ===============================================")
    print("===== FEASOPT =====================================================")
    c.feasopt(c.feasopt.all_constraints())
    print("===================================================================")

    success_statuses = [
        c.solution.status.MIP_feasible,
        c.solution.status.MIP_feasible_relaxed_sum,
    ]

    status = c.solution.get_status()
    status_string = c.solution.get_status_string()
    print("CPLEX STATUS:", status, status_string)

    method = c.solution.get_method()
    print("CPLEX METHOD:", method, c.solution.method[method])

    if status in success_statuses:
        return "success"

    else:
        # TODO: Solution was not MIP optimal. It could be infeasible, suboptimal, etc. Check status.
        return None


def solve(mdp, gamma, constant_state_values=None, relax_infeasible=False, save=False):
    """
    :param mdp: An MDP.
    :param gamma: Learning rate.
    :param constant_state_values: A map {state: value} for state values that we want to use as constants. These are
    states in the mdp that we don't want CPLEX to find a new solution for; their solution is "fixed". Every other state
    not in constant_state_values will be considered a "variable" for CPLEX, and a (new) solution will be generated for
    them.
    :param relax_infeasible: Use CPLEX's feasopt feature to relax the problem to a feasible one.
    :param save: Save all LP's to files.
    :return: None (if no solution was found) or dictionary with keys: "objective_value", "values" and "policy".
    """
    memory_mdp = MemoryMDP(mdp)
    __validate(memory_mdp)

    if constant_state_values is None:
        constant_state_values = {}

    assert all(s in memory_mdp.states for s in constant_state_values)

    c = __create_problem(memory_mdp, gamma, constant_state_values)

    if hasattr(mdp, "name") and save:
        print("Saving LP to file: {}".format(mdp.name))
        Path("scrap-data").mkdir(parents=True, exist_ok=True)
        c.write("scrap-data/{}.lp".format(mdp.name))

    s = __solve_optimally(c)

    if s == "infeasible" and relax_infeasible:
        s = __solve_feasibly(c)

    if s == "success":
        objective_value = c.solution.get_objective_value()
        values = c.solution.get_values()
        policy = __get_policy(values, memory_mdp, gamma, constant_state_values)

        solving_states = []
        for i in range(memory_mdp.n_states):
            state = memory_mdp.states[i]
            if state not in constant_state_values:
                solving_states.append(state)

        assert len(values) == len(solving_states)

        return {
            'objective_value': objective_value,
            'values': {solving_states[i]: value for i, value in enumerate(values)},
            'policy': {solving_states[i]: memory_mdp.actions[j] for i, j in enumerate(policy)}
        }

    return None
