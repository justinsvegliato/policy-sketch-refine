import math
import random

from argparse import ArgumentParser

from termcolor import colored

import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
from simple_abstract_mdp import AbstractMDP
from grid_world_mdp import GridWorldMDP


def main():
    # ------------------ ARGUMENTS ------------------
    parser = ArgumentParser()
    parser.add_argument("method")

    args, other_args = parser.parse_known_args()

    method = args.method
    # ------------------ ARGUMENTS ------------------

    # ------------------ CONSTANTS ------------------
    # If you want to make it more general, move constants to arguments above
    gamma = 0.99
    grid_world_width = 4
    grid_world_height = 4
    wall_probability = 0.30
    random_seed = 5
    tau = 0.10  # Abstract/Ground reduction rate
    # ------------------ CONSTANTS ------------------

    random.seed(random_seed)
    grid_world = utils.generate_random_grid_world(grid_world_width, grid_world_height, wall_probability)

    print("Grid World Domain:")
    printer.print_grid_world_domain(grid_world)

    print()

    print("Setting up the ground MDP...")
    ground_mdp = GridWorldMDP(grid_world)
    # ground_mdp.compute_abstract_states(9)

    if method == "opt":
        for state in ground_mdp.states():
            print(f"\nSTATE {state}:")
            print(f"Transitions:")
            for action in ground_mdp.actions():
                r = ground_mdp.reward_function(state, action)
                print(f"Reward for action {action}: {r}")
                t_sum = 0
                for a in ground_mdp.states():
                    t = ground_mdp.transition_function(state, action, a)
                    if t != 0:
                        t_sum += t
                        print(f"{action}({state},{a}) = {t}")
                print(f"Sum = {t_sum}")

        print("Solving the ground MDP...")
        ground_solution = cplex_mdp_solver.solve(ground_mdp, gamma)
        printer.print_solution(ground_solution)
    
        print("Concrete Grid World Policy:")
        printer.print_grid_world_policy(grid_world, ground_solution['policy'])

        print()

    elif method == "sr":
        print("Setting up the abstract MDP...")

        n_ground_states = len(ground_mdp.states())
        n_required_abstract_states = int(math.ceil(n_ground_states * tau))
        print("Ground states: {}. Required abstract states: {}".format(n_ground_states, n_required_abstract_states))

        abstract_mdp = AbstractMDP(ground_mdp, 'MEAN', n_required_abstract_states)
        print(ground_mdp.states())
        print(abstract_mdp.states())
        for abstract_state in abstract_mdp.states():
            print(f"\nSTATE {abstract_state}:")
            print(f"Ground states: {list(abstract_mdp.get_ground_states([abstract_state]))}")
            print(f"Transitions:")
            for action in abstract_mdp.actions():
                r = abstract_mdp.reward_function(abstract_state, action)
                print(colored(f"Reward for action {action}: {r}", "red"))
                t_sum = 0
                for a in abstract_mdp.states():
                    t = abstract_mdp.transition_function(abstract_state, action, a)
                    if t != 0:
                        t_sum += t
                        print(colored(f"{action}({abstract_state},{a}) = {t}", "blue"))
                print(f"Sum = {t_sum}")

        n_abstract_states = len(abstract_mdp.states())
        print("Obtained abstract states: {}".format(n_abstract_states))

        print("Running the policy-sketch-refine algorithm...")
        for partial_solution in policy_sketch_refine.solve(ground_mdp, abstract_mdp, gamma):
            printer.print_solution(partial_solution)
            partial_policy = utils.get_ground_policy(partial_solution['policy'], ground_mdp, abstract_mdp)
            printer.print_grid_world_policy(grid_world, partial_policy)
            # input("...")

        print("Sketch-Refine Grid World Policy:")
        sketch_refine_policy = utils.get_ground_policy(sketch_refine_solution['policy'], ground_mdp, abstract_mdp)
        printer.print_grid_world_policy(grid_world, sketch_refine_policy)

    else:
        raise Exception("Method '{}' does not exist".format(method))


if __name__ == '__main__':
    main()
