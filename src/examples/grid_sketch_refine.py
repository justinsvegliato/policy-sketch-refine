import math
import random

from argparse import ArgumentParser

import numpy as np
from termcolor import colored

import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
from examples.sample_grid_worlds import *
from simple_abstract_mdp import SimpleAbstractMDP
from complex_abstract_mdp import ComplexAbstractMDP
from grid_world_mdp import GridWorldMDP


def main():
    # ------------------ ARGUMENTS ------------------
    parser = ArgumentParser()
    parser.add_argument("gamma", type=float)
    parser.add_argument("abstraction")

    args, other_args = parser.parse_known_args()

    gamma = args.gamma
    abstraction = args.abstraction
    # ------------------ ARGUMENTS ------------------

    # ------------------ CONSTANTS ------------------
    # If you want to make it more general, move constants to arguments above
    grid_world_width = 15
    grid_world_height = 15
    abstract_grid_width = 5
    abstract_grid_height = 5
    wall_probability = 0.2
    epsilon = 0.9
    abstraction_function = "MEAN"
    random_seed = 6
    tau = 0.10  # Abstract/Ground reduction rate
    relax_infeasible = False
    # ------------------ CONSTANTS ------------------

    random.seed(random_seed)
    # grid_world = utils.generate_random_grid_world(grid_world_width, grid_world_height, wall_probability)
    # grid_world = grid_world_4x4_1
    grid_world = grid_world_10x10_weird_top_left
    # grid_world = grid_world_10x10_weird_top_left_fixed

    print("Grid World Domain:")
    printer.print_grid_world_domain(grid_world)

    print()

    print("Setting up the ground MDP...")
    ground_mdp = GridWorldMDP(grid_world)

    max_reward = max(ground_mdp.reward_function(s, a) for s in ground_mdp.states() for a in ground_mdp.actions())
    print(f"Max reward: {max_reward}")

    input("...")

    #################################################################################
    # Solving Optimally
    #################################################################################
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

    #################################################################################
    # Solving Approximately with SketchRefine
    #################################################################################
    print("Setting up the abstract MDP...")

    n_ground_states = len(ground_mdp.states())
    n_required_abstract_states = int(math.ceil(n_ground_states * tau))
    print("Ground states: {}. Required abstract states: {}".format(n_ground_states, n_required_abstract_states))

    if abstraction == "simple":
        abstract_mdp = SimpleAbstractMDP(ground_mdp, abstraction_function, abstract_grid_width, abstract_grid_height)
    elif abstraction == "complex":
        abstract_mdp = ComplexAbstractMDP(ground_mdp, epsilon, abstraction_function)
    else:
        raise ValueError(abstraction)

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

    max_reward = max(abstract_mdp.reward_function(s, a)
                     for s in abstract_mdp.states()
                     for a in abstract_mdp.actions())
    print(f"Max reward: {max_reward}")

    n_abstract_states = len(abstract_mdp.states())
    print("Obtained abstract states: {}".format(n_abstract_states))

    print("Running the policy-sketch-refine algorithm...")
    sketch_refine_solutions = policy_sketch_refine.solve(
        ground_mdp, abstract_mdp, gamma, relax_infeasible=relax_infeasible)
    for partial_solution in sketch_refine_solutions:
        partial_ground_values = {}
        for row in range(len(grid_world)):
            for column in range(len(grid_world[row])):
                state = len(grid_world[row]) * row + column
                if grid_world[row][column] == 'W':
                    partial_ground_values[state] = 0
                elif state in partial_solution["values"]:
                    partial_ground_values[state] = partial_solution["values"][state]
                else:
                    partial_ground_values[state] = partial_solution["values"][abstract_mdp.get_abstract_state(state)]

        printer.print_solution(partial_solution)

        # Recomputing policy here based on partial_ground_values
        partial_policy = {}
        for state in ground_mdp.states():
            best_action, best_action_value = None, None

            for action in ground_mdp.actions():
                sum_successor_values = np.sum([
                    ground_mdp.transition_function(state, action, succ_state) * partial_ground_values[succ_state]
                    for succ_state in ground_mdp.states()
                ])
                action_value = ground_mdp.reward_function(state, action) + gamma * sum_successor_values
                if best_action_value is None or action_value > best_action_value:
                    best_action = action
                    best_action_value = action_value

            partial_policy[state] = best_action

        print()
        print("SketchRefine Current Policy:")
        printer.print_grid_world_policy(grid_world, partial_policy)

        abs_value_differences = {
            state: abs(ground_solution["values"][state] - partial_ground_values[state])
            for state in ground_mdp.states()
        }
        np.set_printoptions(suppress=True)

        print()
        print("Optimal Values:")
        printer.print_grid_world_values(grid_world, ground_solution["values"])

        print()
        print("SketchRefine Current Values:")
        printer.print_grid_world_values(grid_world, partial_ground_values)

        print()
        print("Absolute Differences:")
        printer.print_grid_world_values(grid_world, abs_value_differences)
        sum_abs_differences = sum(abs_value_differences.values())

        print()
        print(f"Sum of differences: {sum_abs_differences}")


if __name__ == '__main__':
    main()
