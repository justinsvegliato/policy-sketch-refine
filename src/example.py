import cplex_mdp_solver
import printer
import utils
from abstract_mdp import AbstractMDP
from grid_world_mdp import GridWorldMDP


# TODO: Somehow clean up the gross way of printing out policies
def main():
    grid_world = [
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ]

    print("Grid World Domain:")
    printer.print_grid_world_domain(grid_world)

    print()

    print("Setting up the ground mdp...")
    ground_mdp = GridWorldMDP(grid_world)

    print("Solving the ground MDP...")
    solution = cplex_mdp_solver.solve(ground_mdp, 0.99)
    # printer.print_solution(solution)

    print("Concrete Grid World Policy:")
    canonical_policy = [ground_mdp.actions()[value] for value in solution['policy']]
    printer.print_grid_world_policy(grid_world, canonical_policy)

    print()

    print("Setting up the abstract mdp...")
    abstract_mdp = AbstractMDP(ground_mdp, 0.9, 'MEAN')

    print("Solving the abstract MDP...")
    solution = cplex_mdp_solver.solve(abstract_mdp, 0.99)
    # printer.print_solution(solution)

    print("Abstract Grid World Policy:")
    ground_policy = utils.get_ground_policy(solution['policy'], ground_mdp, abstract_mdp)
    canonical_policy = [ground_mdp.actions()[ground_policy[key]] for key in ground_policy]
    printer.print_grid_world_policy(grid_world, canonical_policy)


if __name__ == '__main__':
    main()
