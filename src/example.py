import cplex_mdp_solver
import printer
import utils
from abstract_mdp import AbstractMDP
from grid_world_mdp import GridWorldMDP


def main():
    print("Setting up the grid world mdp...")
    grid_world = [
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ]
    grid_world_mdp = GridWorldMDP(grid_world)

    print("Solving the grid world MDP...")
    solution = cplex_mdp_solver.solve(grid_world_mdp, 0.99)

    print("Objective Value: {:.2f}".format(solution['objective_value']))
    print("Values: {}".format(', '.join('{:.2f}'.format(value) for value in solution['values'])))
    print("Policy: {}".format(', '.join('{}'.format(action) for action in solution['policy'])))

    print("Printing the grid world domain...")
    printer.print_grid_world_domain(grid_world)

    print("Printing the policy...")
    canonical_policy = [grid_world_mdp.actions()[value] for value in solution['policy']]
    printer.print_grid_world_policy(grid_world, canonical_policy)

    print("Setting up the abstract mdp...")
    abstract_mdp = AbstractMDP(grid_world_mdp, 0.89, 'MEAN')

    print(abstract_mdp.abstract_states)

    print("Solving the abstract MDP...")
    solution = cplex_mdp_solver.solve(abstract_mdp, 0.99)

    print("Objective Value: {:.2f}".format(solution['objective_value']))
    print("Values: {}".format(', '.join('{:.2f}'.format(value) for value in solution['values'])))
    print("Policy: {}".format(', '.join('{}'.format(action) for action in solution['policy'])))

    print("Printing the policy...")
    ground_policy = utils.get_ground_policy(solution['policy'], grid_world_mdp, abstract_mdp)
    canonical_policy = [grid_world_mdp.actions()[ground_policy[key]] for key in ground_policy.keys()]
    printer.print_grid_world_policy(grid_world, canonical_policy)


if __name__ == '__main__':
    main()
