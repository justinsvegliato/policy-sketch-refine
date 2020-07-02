import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
from simple_abstract_mdp import SimpleAbstractMDP
from grid_world_mdp import GridWorldMDP
from partially_abstract_mdp import PartiallyAbstractMDP


# TODO: Somehow clean up the gross way of printing out policies
def main():
    # grid_world = [
    #     ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
    #     ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
    #     ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
    #     ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
    #     ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    #     ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    # ]

    grid_world = utils.generate_random_grid_world(10, 10, 0.05)

    print("Grid World Domain:")
    printer.print_grid_world_domain(grid_world)

    # print()

    print("Setting up the ground MDP...")
    ground_mdp = GridWorldMDP(grid_world)

    # print("Solving the ground MDP...")
    # ground_solution = cplex_mdp_solver.solve(ground_mdp, 0.99)
    # printer.print_solution(solution)

    # print("Concrete Grid World Policy:")
    # printer.print_grid_world_policy(grid_world, ground_solution['policy'])

    # print()

    print("Setting up the abstract MDP...")
    abstract_mdp = SimpleAbstractMDP(ground_mdp, 'MEAN', 3, 3)

    # print("Solving the abstract MDP...")
    # abstract_solution = cplex_mdp_solver.solve(abstract_mdp, 0.99)
    # printer.print_solution(abstract_solution)

    # print("Abstract Grid World Policy:")
    # ground_policy = utils.get_ground_policy(abstract_solution['policy'], ground_mdp, abstract_mdp)
    # printer.print_grid_world_policy(grid_world, ground_policy)

    # print()

    print("Setting up the partially abstract MDP...")
    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, ['abstract_0', 'abstract_1'])

    # print("Solving the partially abstract MDP...")
    # partially_abstract_solution = cplex_mdp_solver.solve(partially_abstract_mdp, 0.99)
    # printer.print_solution(solution)

    # print("Partially Abstract Grid World Policy:")
    # ground_policy = utils.get_ground_policy(partially_abstract_solution['policy'], ground_mdp, abstract_mdp)
    # printer.print_grid_world_policy(grid_world, ground_policy)

    # print()

    print("Running the policy-sketch-refine algorithm...")
    sketch_refine_solution = policy_sketch_refine.solve(ground_mdp, 0.99)

    print("Sketch-Refine Grid World Policy:")
    sketch_refine_policy = utils.get_ground_policy(sketch_refine_solution['policy'], ground_mdp, abstract_mdp)
    printer.print_grid_world_policy(grid_world, sketch_refine_policy)


if __name__ == '__main__':
    main()
