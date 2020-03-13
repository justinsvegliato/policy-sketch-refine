import numpy as np

from memory_mdp import MemoryMDP
from grid_world_mdp import GridWorldMDP
from overrides import overrides
import cplex


class CplexMDP(MemoryMDP):
    def __init__(self):
        super().__init__()
        self.c = None  # The CPLEX problem

    @overrides
    def formulate_lp(self, gamma):
        super(CplexMDP, self).formulate_lp()

        # Create empty CPLEX problem
        self.c = cplex.Cplex()
        print("CPLEX {}".format(self.c.get_version()))

        # There is a continuous decision variable for each state, i.e., its "value"
        self.c.variables.add(types=[self.c.variables.type.continuous] * self.n_states)
        print("{} variables".format(self.c.variables.get_num()))

        # ========= Objective function =========
        self.c.objective.set_linear(
            [(i, self.start_probabilities[i]) for i in range(self.n_states)])
        self.c.objective.set_sense(self.c.objective.sense.minimize)

        # ========= Linear constraints =========
        lin_exp = []
        rhs = []

        # Each constraint will use all states' variables (as the "next possible states")
        variables = range(self.n_states)

        # There is one constraint for each (state, action) pair
        for i in range(self.n_states):
            for j in range(self.n_actions):
                coefficients = []
                # Each constraint refers to all state variables (as the "next possible states")
                # Each coefficient depends on whether the next possible state is the current state or not
                for k in range(self.n_states):
                    # If the next possible state is not the current state
                    if k != i:
                        coefficient = - gamma * self.transition_probabilities[i, j, k]
                    # If the next possible is the current state
                    else:
                        coefficient = 1 - gamma * self.transition_probabilities[i, j, k]
                    coefficients.append(coefficient)

                # Append linear constraint
                lin_exp.append([variables, coefficients])

                # The constraint's right-hand side is simply the reward
                rhs.append(float(self.rewards[i, j]))

                print(coefficients, self.rewards[i, j])

        # Add *all* linear constraints to CPLEX *at once*
        self.c.linear_constraints.add(lin_expr=lin_exp, rhs=rhs, senses=["G"]*len(rhs))

        print("{} linear constraints".format(self.c.linear_constraints.get_num()))
        print("linear objective: {}".format(self.c.objective.get_linear()))
        self.c.write("mdp.lp")

    @overrides
    def solve_lp(self):
        assert isinstance(self.c, cplex.Cplex)
        self.c.solve()
        # print(self.c.solution.get_status_string())
        # print(self.c.solution.get_values())

    @overrides
    def get_solution(self, gamma):
        assert isinstance(self.c, cplex.Cplex)

        state_values = self.c.solution.get_values()

        policy = []
        for i in range(self.n_states):
            best_action, best_action_value = None, None
            for j in range(self.n_actions):
                action_value = self.rewards[i, j] + gamma * np.sum(self.transition_probabilities[i, j] * state_values)
                if best_action_value is None or action_value > best_action_value:
                    best_action = j
                    best_action_value = action_value
            policy.append(best_action)

        return {
            "objective_value": self.c.solution.get_objective_value(),
            "state_values": state_values,
            "policy": policy,
        }
