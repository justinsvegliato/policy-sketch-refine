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
    s = mdp.get_solution(gamma=0.9)

    print()

    print(s['objective_value'])
    print(s['state_values'])
    print(s['policy'])

    print()

    print('Printing the grid world domain...')
    printer.print_grid_world_domain(grid_world)

    print()

    print('Printing the policy...')
    canonical_policy = [grid_world_mdp.actions()[value] for value in s["policy"]]
    printer.print_grid_world_policy(grid_world, canonical_policy)


if __name__ == '__main__':
    main()
