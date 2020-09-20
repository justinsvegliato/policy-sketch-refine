import cplex
import numpy as np

IS_VERBOSE = False
IS_RECORDING = False

LOWER_BOUND = -10000
UPPER_BOUND = 10000


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


def validate(memory_mdp, constant_state_values):
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

    assert all(state in memory_mdp.states for state in constant_state_values)


# TODO: Determine if we need lower and upper bounds in this function
def set_variables(problem, memory_mdp, constant_state_values):
    n_variable_states = memory_mdp.n_states - len(constant_state_values)

    if IS_VERBOSE:
        print("Variable State Count: {}".format(n_variable_states))

    types = [problem.variables.type.continuous] * n_variable_states
    lower_bound = [LOWER_BOUND] * n_variable_states
    upper_bound = [UPPER_BOUND] * n_variable_states

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
    linear_expressions = []
    right_hand_sides = []
    names = []

    n_variable_states = memory_mdp.n_states - len(constant_state_values)
    variables = range(n_variable_states)

    # Create a linear constraint for each state-action pair
    for i in range(memory_mdp.n_states):
        for j in range(memory_mdp.n_actions):
            # Define a linear constraint for the state-action pair
            right_hand_side = memory_mdp.rewards[i, j]

            state = memory_mdp.states[i]

            # Discount its value from the right hand side of the constraint if the start state is a constant
            if state in constant_state_values:
                right_hand_side -= constant_state_values[state]

            # Loop across all of the existing states as successor states by either:
            # (a) setting the coefficients of the variable to correspond to variables states
            # (b) modifying the right hand side for constant states
            coefficients = []
            for k in range(memory_mdp.n_states):
                successor_state = memory_mdp.states[k]

                # Use the value and the transition probability of the successor state to modify the right hand side of the constraint
                if successor_state in constant_state_values:
                    right_hand_side += gamma * memory_mdp.transition_probabilities[i, j, k] * constant_state_values[successor_state]
                # Set the coefficient of the successor state's variable
                else:
                    # Check if the successor state is not the start state
                    if k != i:
                        coefficient = - gamma * memory_mdp.transition_probabilities[i, j, k]
                    # Check if the successor state is the start state
                    else:
                        coefficient = 1 - gamma * memory_mdp.transition_probabilities[i, j, k]
                    coefficients.append(coefficient)

            # TODO: Determine why this problem happens
            if sum(coefficients) <= 0 < right_hand_side:
                continue

            # Skip useless constraints without any variables
            if all(coefficient == 0 for coefficient in coefficients):
                continue

            # TODO: Avoid using 0 coefficients altogether if necessary for space complexity
            # TODO: Do we need to cast the right hand side as a float - seems not needed
            linear_expressions.append([variables, coefficients])
            right_hand_sides.append(float(right_hand_side))
            names.append(f"{state}_{memory_mdp.actions[j]}")

    senses = ["G"] * len(linear_expressions)

    # Add all linear constraints to CPLEX at once
    problem.linear_constraints.add(names=names, lin_expr=linear_expressions, rhs=right_hand_sides, senses=senses)


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

    validate(memory_mdp, constant_state_values)

    problem = create_problem(memory_mdp, gamma, constant_state_values)

    if IS_RECORDING and hasattr(mdp, 'name'):
        print(f"Saving the problem to the file mdp-{mdp.name}.lp...")
        problem.write(f'logs/mdp-{mdp.name}.lp')

    status = solve_optimally(problem)

    if status == 'INFEASIBLE' and relax_infeasible:
        status = solve_feasibly(problem)

    if status == 'SUCCESS':
        objective_value = problem.solution.get_objective_value()
        values = problem.solution.get_values()
        policy = get_policy(values, memory_mdp, gamma, constant_state_values)

        # TODO: Clean up all of this stuff
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
