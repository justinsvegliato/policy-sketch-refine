from grid_world_mdp import GridWorldMDP
from cplex_mdp import CplexMDP


def main():
    grid_world_mdp = GridWorldMDP([
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ])

    print("Solving the grid world MDP...")
    mdp = CplexMDP()
    mdp.load_mdp(grid_world_mdp)
    mdp.formulate_lp(gamma=0.9)
    mdp.solve_lp()
    s = mdp.get_solution(gamma=0.9)

    print(s["objective_value"])
    print(s["state_values"])
    print(s["policy"])

    # print("Solving a random MDP...")
    # mdp = CplexMDP()
    # mdp.load_random_mdp(500, 2)
    # mdp.formulate_lp(gamma=0.5)
    # mdp.solve_lp()


if __name__ == '__main__':
    main()
