from termcolor import colored

import cplex_mdp_solver
from memory_mdp import MemoryMDP
from partially_abstract_mdp import PartiallyAbstractMDP


def __sketch(abstract_mdp, gamma, relax_infeasible):
    abstraction = "MEAN"
    # abstract_mdp = AbstractMDP(ground_mdp, abstraction, n_abstract_states)

    # print("Ground MPD has {} states".format(len(ground_mdp.states())))
    print("Abstract MPD (abstraction={}) has {} states".format(
        abstraction,
        len(abstract_mdp.states())))
    input("Continue?...")
    abstract_mdp.name = "abs"
    sketch = cplex_mdp_solver.solve(abstract_mdp, gamma, relax_infeasible=relax_infeasible)
    return sketch


def __iterative_refine(ground_mdp, abstract_mdp, refined_ground_values, gamma, relax_infeasible):
    # -------------------------------------------------------------------------------
    # Phase 1 - Do all refines independently (in parallel) and then combine solutions
    # -------------------------------------------------------------------------------

    all_abstract_states = set(abstract_mdp.states())

    # Note: With Iterative-Refine, the refining order does not matter
    refining_abstract_states = all_abstract_states.copy()

    iteration_number = 1

    while refining_abstract_states:
        print("PHASE 1 - ITERATION {}".format(iteration_number))
        input("Press key...")

        refined_abstract_states = all_abstract_states - refining_abstract_states

        refines = []

        for refining_abstract_state in sorted(refining_abstract_states):
            print()
            print("Refining {}".format(refining_abstract_state))

            # Abstract MDP where some abstract states are replaced with ground states
            # (the remaining states are all abstract).
            # Note: From iteration 2 on, this will also contain ground states for abstract states already refined.
            grounding_abstract_states = refined_abstract_states | {refining_abstract_state}

            refining_ground_states = abstract_mdp.get_ground_states([refining_abstract_state])

            print("Refining abstract states: {}".format(sorted(refining_abstract_states)))
            print("Grounding abstract states: {}".format(sorted(grounding_abstract_states)))
            print("Refining ground states: {}".format(refining_ground_states))
            print("Creating PAMDP ({} refining ground states)...".format(len(refining_ground_states)))
            partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, grounding_abstract_states)

            for a1 in partially_abstract_mdp.states():
                print(f"\nSTATE {a1}:")
                for act in partially_abstract_mdp.actions():
                    r = partially_abstract_mdp.reward_function(a1, act)
                    print(colored(f"Reward for action {act}: {r}", "red"))
                    print(f"Transitions for action {act}:")
                    t_sum = 0
                    for a2 in partially_abstract_mdp.states():
                        t = partially_abstract_mdp.transition_function(a1, act, a2)
                        if t != 0:
                            print(colored(f"{act}({a1},{a2}) = {t}", "blue"))
                            t_sum += t
                    print(f"Sum = {t_sum}")
                    assert round(t_sum, 6) == 1

            # Set as constant all states that are not the current "solving abstract state"
            constant_state_values = {}
            for state in partially_abstract_mdp.states():
                # FIXME: Better way to distinguish abstract from ground states?
                if type(state) is str and state.startswith("abstract"):
                    # This abstract state was never refined
                    if state != refining_abstract_state:
                        # For an unrefined abstract state, we take its abstract solution's value (sketch solution)
                        ground_state = abstract_mdp.get_ground_states([state])[0]  # Any ground state
                        constant_state_values[state] = refined_ground_values[ground_state]
                        # constant_state_values[state] = sketch["values"][state]
                else:
                    # This ground state was already refined
                    corresponding_abstract_state = abstract_mdp.get_abstract_state(state)
                    if corresponding_abstract_state != refining_abstract_state:
                        # We're not refining it again this time: we add to the constants its ground value
                        constant_state_values[state] = refined_ground_values[state]

            print("Constant states: {}".format(constant_state_values.keys()))
            print("Constant values: {}".format(constant_state_values.items()))

            # input("Press to solve...")

            # Solve this MDP: find values for
            partially_abstract_mdp.name = "it{}_ref{}".format(iteration_number, refining_abstract_state)
            partially_abstract_solution = cplex_mdp_solver.solve(
                partially_abstract_mdp, gamma, constant_state_values, relax_infeasible=relax_infeasible)

            # input("Continue?...")

            # TODO: These MDPs can grow very large. When scaling this up, don't store them all in main memory like this
            # TODO: It's okay to keep one MDP at a time in main memory.
            refines.append((refining_abstract_state, partially_abstract_mdp, partially_abstract_solution))

        n_successful_refines = sum(1 for a, b, c in refines if c is not None)

        if n_successful_refines == 0:
            raise Exception("Can't make progress in Phase 1!")
            # Refine all again
            # refining_abstract_states = all_abstract_states.copy()

        else:
            # Now integrate solutions
            for refining_abstract_state, partially_abstract_mdp, partially_abstract_solution in refines:
                refining_ground_states = abstract_mdp.get_ground_states([refining_abstract_state])

                if partially_abstract_solution is not None:
                    print("Partially Abstract objective value: {}".format(partially_abstract_solution["objective_value"]))
                    # for ground_state in ground_mdp.states():
                    # if ground_state in partially_abstract_solution["values"]:
                    for ground_state in refining_ground_states:
                        print("Updating ground value of state {} to {}".format(
                            ground_state, partially_abstract_solution["values"][ground_state]))
                        refined_ground_values[ground_state] = partially_abstract_solution["values"][ground_state]

                    # Set abstract state as "refined"
                    refining_abstract_states -= {refining_abstract_state}

                # FIXME: This is a quick & dirty implementation (should not recreate ground memory mdp here)
                memory_mdp = MemoryMDP(ground_mdp)
                values = [refined_ground_values[memory_mdp.states[i]] for i in range(memory_mdp.n_states)]
                policy = cplex_mdp_solver.__get_policy(values, memory_mdp, gamma)
                yield {
                    "objective_value": float("nan"),  # FIXME: Compute and return objective value
                    "values": refined_ground_values,
                    "policy": {memory_mdp.states[i]: memory_mdp.actions[j] for i, j in enumerate(policy)},
                }

        # Refine all again
        # refining_abstract_states = all_abstract_states.copy()

        print("N successful refines: {} out of {}".format(n_successful_refines, len(refines)))

        iteration_number += 1

    # TODO: Phase 2 - Check feasibility and adjust if infeasible

    print("Refined Ground Values at the end:")
    print(refined_ground_values)

    # FIXME: This is a quick & dirty implementation (should not recreate ground memory mdp here)
    memory_mdp = MemoryMDP(ground_mdp)
    values = [refined_ground_values[memory_mdp.states[i]] for i in range(memory_mdp.n_states)]
    ground_policy = cplex_mdp_solver.__get_policy(values, memory_mdp, gamma)
    yield {
        "objective_value": float("nan"),  # FIXME: Compute and return objective value
        "values": refined_ground_values,
        "policy": {memory_mdp.states[i]: memory_mdp.actions[j] for i, j in enumerate(ground_policy)},
    }


def solve(ground_mdp, abstract_mdp, gamma, relax_infeasible):
    sketch = __sketch(abstract_mdp, gamma, relax_infeasible)
    yield sketch

    # Combined solution
    # Initializing every ground state value with its corresponding abstract solution's state value
    refined_ground_values = {}
    for abstract_state, ground_states in abstract_mdp.abstract_states.items():
        for ground_state in ground_states:
            assert ground_state not in refined_ground_values
            refined_ground_values[ground_state] = sketch["values"][abstract_state]

    for partially_refined in __iterative_refine(
            ground_mdp, abstract_mdp, refined_ground_values, gamma, relax_infeasible):
        yield partially_refined
