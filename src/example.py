from grid_world_mdp import GridWorldMDP
from cplex_mdp import CplexMDP
import printer


def main():
    grid_world = [
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ]
    grid_world_mdp = GridWorldMDP(grid_world)

    print('Solving the grid world MDP...')
    mdp = CplexMDP()
    mdp.load_mdp(grid_world_mdp)
    mdp.formulate_lp(gamma=0.9)
    mdp.solve_lp()
    solution = mdp.get_solution(gamma=0.9)

    print("Objective Value: {:.2f}".format(solution['objective_value']))
    print("State Values: {}".format(", ".join("{:.2f}".format(value) for value in solution['state_values'])))
    print("Policy: {}".format(", ".join("{}".format(action) for action in solution['policy'])))

    print('Printing the grid world domain...')
    printer.print_grid_world_domain(grid_world)

    print('Printing the policy...')
    canonical_policy = [grid_world_mdp.actions()[value] for value in solution['policy']]
    printer.print_grid_world_policy(grid_world, canonical_policy)


if __name__ == '__main__':
    main()
