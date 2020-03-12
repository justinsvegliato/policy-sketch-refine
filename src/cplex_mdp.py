from src.memory_mdp import MemoryMDP
from src.grid_world_mdp import GridWorldMDP
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
        print(self.c.get_version())

        # There is continuous decision variable for each state, i.e., its "value"
        self.c.variables.add(types=[self.c.variables.type.continuous] * self.n_states)
        print(self.c.variables.get_num(), "variables")

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
                lin_exp.append(cplex.SparsePair(ind=variables, val=coefficients))

                # The constraint's right-hand side is simply the reward
                rhs.append(float(self.rewards[i, j]))

        # All *all* linear constraints to CPLEX *at once*
        self.c.linear_constraints.add(lin_expr=lin_exp, rhs=rhs, senses=["G"]*len(rhs))

    @overrides
    def solve_lp(self):
        assert isinstance(self.c, cplex.Cplex)
        self.c.solve()
        print(self.c.solution.get_status_string())
        print(self.c.solution.get_values())


if __name__ == "__main__":
    grid_world_mdp = GridWorldMDP([
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ])

    mdp = CplexMDP()
    mdp.load_mdp(grid_world_mdp)
    mdp.formulate_lp(gamma=0.9)
    mdp.solve_lp()
