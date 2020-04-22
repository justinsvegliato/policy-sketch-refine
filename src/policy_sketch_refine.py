import cplex_mdp_solver
from simple_abstract_mdp import AbstractMDP
from memory_mdp import MemoryMDP
from partially_abstract_mdp import PartiallyAbstractMDP


def __sketch(ground_mdp, gamma):
    abstraction = "MEAN"
    abstract_mdp = AbstractMDP(ground_mdp, abstraction)

    print("Ground MPD has {} states".format(len(ground_mdp.states())))
    print("Abstract MPD (abstraction={}) has {} states".format(
        abstraction,
        len(abstract_mdp.states())))
    input("Continue?...")
    sketch = cplex_mdp_solver.solve(abstract_mdp, gamma)
    return abstract_mdp, sketch


def __iterative_refine(ground_mdp, abstract_mdp, sketch, gamma):
    # -------------------------------------------------------------------------------
    # Phase 1 - Do all refines independently (in parallel) and then combine solutions
    # -------------------------------------------------------------------------------

    # Combined solution
    # Initializing every ground state value with its corresponding abstract solution's state value
    refined_ground_values = {}
    for abstract_state, ground_states in abstract_mdp.abstract_states.items():
        for ground_state in ground_states:
            assert ground_state not in refined_ground_values
            refined_ground_values[ground_state] = sketch["values"][abstract_state]

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
            partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, grounding_abstract_states)

            # Set as constant all states that are not the current "solving abstract state"
            constant_state_values = {}
            for state in partially_abstract_mdp.states():
                # FIXME: Better way to distinguish abstract from ground states?
                if type(state) is str and state.startswith("abstract"):
                    # This abstract state was never refined
                    if state != refining_abstract_state:
                        # For an unrefined abstract state, we take its abstract solution's value (sketch solution)
                        constant_state_values[state] = sketch["values"][state]
                else:
                    # This ground state was already refined
                    corresponding_abstract_state = abstract_mdp.get_abstract_state(state)
                    if corresponding_abstract_state != refining_abstract_state:
                        # We're not refining it again this time: we add to the constants its ground value
                        constant_state_values[state] = refined_ground_values[state]

            # Solve this MDP: find values for
            partially_abstract_mdp.name = "{}_{}".format(iteration_number, refining_abstract_state)
            partially_abstract_solution = cplex_mdp_solver.solve(partially_abstract_mdp, 0.99, constant_state_values)

            # TODO: These MDPs can grow very large. When scaling this up, don't store them all in main memory like this
            # TODO: It's okay to keep one MDP at a time in main memory.
            refines.append((partially_abstract_mdp, partially_abstract_solution))

            if partially_abstract_solution is not None:
                for ground_state in ground_mdp.states():
                    if ground_state in partially_abstract_solution["values"]:
                        refined_ground_values[ground_state] = partially_abstract_solution["values"][ground_state]

                # Set abstract state as "refined"
                refining_abstract_states -= {refining_abstract_state}

        n_successful_refines = sum(1 for a, b in refines if b is not None)
        print("N successful refines: {} out of {}".format(n_successful_refines, len(refines)))

        if n_successful_refines == 0:
            print("Can't make progress!")
            break

        iteration_number += 1

    # TODO: Phase 2 - Check feasibility and adjust if infeasible

    # FIXME: This is a quick & dirty implementation (should not recreate ground memory mdp here)
    memory_mdp = MemoryMDP(ground_mdp)
    values = [refined_ground_values[memory_mdp.states[i]] for i in range(memory_mdp.n_states)]
    ground_policy = cplex_mdp_solver.__get_policy(values, memory_mdp, gamma)
    return {
        "objective_value": None,  # FIXME
        "values": refined_ground_values,
        "policy": {memory_mdp.states[i]: memory_mdp.actions[j] for i, j in enumerate(ground_policy)},
    }


def solve(ground_mdp, gamma):
    abstract_mdp, sketch = __sketch(ground_mdp, gamma)
    return __iterative_refine(ground_mdp, abstract_mdp, sketch, gamma)

