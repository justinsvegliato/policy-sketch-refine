import cplex
import numpy as np

IS_VERBOSE = False
IS_RECORDING = False


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


def validate(memory_mdp):
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


# TODO: Determine if we need lower and upper bounds in this function
def set_variables(problem, memory_mdp, constant_state_values):
    n_variable_states = memory_mdp.n_states - len(constant_state_values)

    if IS_VERBOSE:
        print("Variable State Count: {}".format(n_variable_states))

    types = [problem.variables.type.continuous] * n_variable_states
    lower_bound = [-10000] * n_variable_states
    upper_bound = [10000] * n_variable_states

    problem.variables.add(types=types, lb=lower_bound, ub=upper_bound)


def set_objective(problem, memory_mdp, constant_state_values):
    n_variable_states = memory_mdp.n_states - len(constant_state_values)

    variable_state_coefficients = []
    for i in range(memory_mdp.n_states):
        state = memory_mdp.states[i]
        if state not in constant_state_values:
            variable_state_coefficients.append(memory_mdp.start_state_probabilities[i])

    if IS_VERBOSE:
        print("Constant States: {}".format(constant_state_values.keys()))
        print("Variable State Coefficients:", variable_state_coefficients)
        print("Variable State Count: {}".format(n_variable_states))

    assert len(variable_state_coefficients) == n_variable_states

    problem.objective.set_linear(enumerate(variable_state_coefficients))
    problem.objective.set_sense(problem.objective.sense.minimize)


def set_constraints(problem, memory_mdp, gamma, constant_state_values):
    lin_expressions = []
    right_hand_sides = []
    names = []

    n_variable_states = memory_mdp.n_states - len(constant_state_values)
    variables = range(n_variable_states)

    # Create one constraint for each (start_state, action) pair
    for i in range(memory_mdp.n_states):
        for j in range(memory_mdp.n_actions):
            # Define 1 linear constraint for a (start_state, action) pair
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
    problem.linear_constraints.add(
        names=names,
        lin_expr=lin_expressions,
        rhs=right_hand_sides,
        senses=["G"] * len(lin_expressions))


def get_policy(values, memory_mdp, gamma, constant_state_values=None):
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


def create_problem(memory_mdp, gamma, constant_state_values):
    problem = cplex.Cplex()

    if not IS_VERBOSE:
        problem.set_log_stream(None)
        problem.set_results_stream(None)

    set_variables(problem, memory_mdp, constant_state_values)
    set_objective(problem, memory_mdp, constant_state_values)
    set_constraints(problem, memory_mdp, gamma, constant_state_values)

    if IS_VERBOSE:
        print("Variable Count:", problem.variables.get_num())
        print("Sense: ", problem.objective.sense[problem.objective.get_sense()])
        print("Linear Coefficients", len(problem.objective.get_linear()))
        print("Linear Constraints", format(problem.linear_constraints.get_num()))
        print("Integer Variable Count:", problem.variables.get_num_integer())
        print("Variable Lower Bounds:", problem.variables.get_lower_bounds())
        print("Variable Upper Bounds:", problem.variables.get_upper_bounds())

    return problem


def solve_optimally(problem):
    problem.solve()

    success_statuses = [problem.solution.status.MIP_optimal, problem.solution.status.optimal_tolerance]
    infeasible_statuses = [problem.solution.status.infeasible, problem.solution.status.MIP_infeasible]

    status = problem.solution.get_status()

    if IS_VERBOSE:
        print("CPLEX Status:", problem.solution.get_status_string())
        print("CPLEX Method:", problem.solution.get_method())

    if status in success_statuses:
        return 'SUCCESS'

    if status in infeasible_statuses:
        return 'INFEASIBLE'

    # TODO: Address the case when the solution is not MIP optimal - could be infeasible or suboptimal
    return None


def solve_feasibly(problem):
    problem.feasopt(problem.feasopt.all_constraints())

    success_statuses = [problem.solution.status.MIP_feasible, problem.solution.status.MIP_feasible_relaxed_sum]

    status = problem.solution.get_status()

    if IS_VERBOSE:
        print("CPLEX Status:", problem.solution.get_status_string())
        print("CPLEX Method:", problem.solution.get_method())

    if status in success_statuses:
        return 'SUCCESS'

    # TODO: Address the case when the solution is not MIP optimal - could be infeasible or suboptimal
    return None


def solve(mdp, gamma, constant_state_values={}, relax_infeasible=False):
    memory_mdp = MemoryMDP(mdp)

    validate(memory_mdp)

    # TODO: Clean this up a little more
    assert all(state in memory_mdp.states for state in constant_state_values)

    problem = create_problem(memory_mdp, gamma, constant_state_values)

    if IS_RECORDING and hasattr(mdp, 'name'):
        print("Saving the linar program to file...")
        problem.write(f'logs/mdp-{mdp.name}.lp')

    status = solve_optimally(problem)

    if status == 'INFEASIBLE' and relax_infeasible:
        status = solve_feasibly(problem)

    # TODO: Clean this up a little more
    if status == 'SUCCESS':
        objective_value = problem.solution.get_objective_value()
        values = problem.solution.get_values()
        policy = get_policy(values, memory_mdp, gamma, constant_state_values)

        variable_states = []
        for i in range(memory_mdp.n_states):
            state = memory_mdp.states[i]
            if state not in constant_state_values:
                variable_states.append(state)

        assert len(values) == len(variable_states)

        return {
            'objective_value': objective_value,
            'values': {variable_states[i]: value for i, value in enumerate(values)},
            'policy': {variable_states[i]: memory_mdp.actions[j] for i, j in enumerate(policy)}
        }

    return None
