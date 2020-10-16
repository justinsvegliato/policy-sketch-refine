import logging
import os
import pickle
import time
import yaml
from termcolor import colored

import cplex_mdp_solver
import policy_sketch_refine
import printer
import utils
from argparse import ArgumentParser
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

# FIXME: Should we change/randomize this one?
INITIAL_GROUND_STATE = 0

logging.basicConfig(
    format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)


def readable_time(time_seconds):
    m, s = divmod(time_seconds, 60)
    h, m = divmod(m, 60)
    return "{:d}:{:02d}:{:02d}".format(int(h), int(m), int(s))


def main():
    # =========================================================================================
    # Set args
    # =========================================================================================
    arg_parser = ArgumentParser()

    # Data Folder (use a partition with enough space)
    arg_parser.add_argument("-d", "--data-dir", required=True)

    # Earth Observation
    arg_parser.add_argument("-W", "--width", required=True, type=int)
    arg_parser.add_argument("-H", "--height", required=True, type=int)
    arg_parser.add_argument("-I", "--n-points-of-interest", required=True, type=int)
    arg_parser.add_argument("-V", "--visibility")
    arg_parser.add_argument("-v", "--random-variation", required=True, type=int)

    # Abstraction
    arg_parser.add_argument("-aA", "--abstract-aggregate", required=True)
    arg_parser.add_argument("-aW", "--abstract-width", required=True, type=int)
    arg_parser.add_argument("-aH", "--abstract-height", required=True, type=int)

    # Simulator
    arg_parser.add_argument("-T", "--time-horizon", required=True, type=int)
    arg_parser.add_argument("-s", "--sleep-duration", required=True, type=float)
    arg_parser.add_argument("--gamma", required=True, type=float)
    arg_parser.add_argument("--expand-poi", required=True)
    arg_parser.add_argument("-f", "--force", default="")

    # =========================================================================================
    # Read args
    # =========================================================================================
    args = arg_parser.parse_args()
    random_variation = args.random_variation
    width = args.width
    height = args.height
    points_of_interest = args.n_points_of_interest
    visibility = args.visibility
    sleep_duration = args.sleep_duration
    time_horizon = args.time_horizon
    data_dir = args.data_dir
    abstract_aggregate = args.abstract_aggregate
    abstract_width = args.abstract_width
    abstract_height = args.abstract_height
    gamma = args.gamma
    expand_poi = args.expand_poi.lower() in ("1", "yes", "y", "t", "true")
    force = args.force.lower() in ("1", "yes", "y", "t", "true")

    # =========================================================================================
    # Run
    # =========================================================================================
    run(data_dir,
        random_variation, width, height, points_of_interest, visibility,
        abstract_aggregate, abstract_width, abstract_height,
        sleep_duration, time_horizon, gamma, expand_poi,
        simulate=True,
        force=force)


def run(data_dir,
        random_variation, width, height, points_of_interest, visibility,
        abstract_aggregate, abstract_width, abstract_height,
        sleep_duration, time_horizon, gamma, expand_poi,
        simulate, force=False):
    size = width, height

    domain_name = f"Earth_Observation_W{width}_H{height}_I{points_of_interest}_V{visibility}_v{random_variation}"
    abstraction_name = f"Abstraction_A{abstract_aggregate}_W{abstract_width}_H{abstract_height}"
    simulation_name = f"Simulation_s{sleep_duration}_T{time_horizon}_gamma{gamma}_Expand{expand_poi}"

    # Check data dir
    if not os.path.isdir(data_dir):
        raise Exception(f"Data directory {data_dir} does not exist. Create it and run this again.")

    # Sub-dirs
    if not os.path.isdir(os.path.join(data_dir, domain_name)):
        os.mkdir(os.path.join(data_dir, domain_name))
    if not os.path.isdir(os.path.join(data_dir, domain_name, abstraction_name)):
        os.mkdir(os.path.join(data_dir, domain_name, abstraction_name))
    if not os.path.isdir(os.path.join(data_dir, domain_name, abstraction_name, simulation_name)):
        os.mkdir(os.path.join(data_dir, domain_name, abstraction_name, simulation_name))

    # Generate Earth Observation MDP
    start = time.time()
    utils.set_random_variation(random_variation)
    ground_mdp = EarthObservationMDP(size, points_of_interest, visibility)
    end = time.time()
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]",
                 len(ground_mdp.states()),
                 len(ground_mdp.actions()),
                 end - start)
    log = {
        "Earth Observation Ground MDP": {
            "Width": width,
            "Height": height,
            "POIs": points_of_interest,
            "Visibility": visibility,
            "Random Variation": random_variation,
            "Number of States": len(ground_mdp.states()),
            "Number of Actions": len(ground_mdp.actions()),
            "Creation Time": round(end - start, 2),
            "Creation Human Time": readable_time(end - start),
        }
    }

    abstract_mdp_file_path = os.path.join(data_dir, domain_name, abstraction_name)

    # Solve the ground MDP only (no abstraction)
    if abstract_aggregate == "NONE":
        start = time.time()
        solution = cplex_mdp_solver.solve(ground_mdp, gamma)
        end = time.time()
        log["Earth Observation Ground MDP"]["Solving Time"] = round(end - start, 2)
        log["Earth Observation Ground MDP"]["Solving Human Time"] = readable_time(end - start)
        yaml.dump(log, open(abstract_mdp_file_path + ".yaml", "w"))
        return

    # Abstract MDP
    if os.path.isfile(abstract_mdp_file_path + ".pickle") and os.path.isfile(abstract_mdp_file_path + ".yaml"):
        print(colored("Abstraction was already done.", "blue"))
        abstract_mdp = pickle.load(open(abstract_mdp_file_path + ".pickle", "rb"))
        log = yaml.load(open(abstract_mdp_file_path + ".yaml"), Loader=yaml.FullLoader)
    elif not simulate:
        start = time.time()
        abstract_mdp = EarthObservationAbstractMDP(ground_mdp, abstract_aggregate, abstract_width, abstract_height)
        end = time.time()

        logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]",
                     len(abstract_mdp.states()),
                     len(abstract_mdp.actions()),
                     end - start)

        # Store abstract MDP
        with open(abstract_mdp_file_path + ".pickle", "wb") as f:
            print(pickle.dump(abstract_mdp, f, pickle.HIGHEST_PROTOCOL))

        # Store abstraction logs
        log["Abstract MDP"] = {
            "Abstraction Aggregate": abstract_aggregate,
            "Abstraction Width": abstract_width,
            "Abstraction Height": abstract_height,
            "Abstraction Time": round(end - start, 2),
            "Abstraction Human Time": readable_time(end - start),
        }
        yaml.dump(log, open(abstract_mdp_file_path + ".yaml", "w"))
    else:
        raise AssertionError("No abstraction found in data folder but you wanted to simulate now.")

    if not simulate:
        return

    # ==============================================================================
    # Simulator
    # ==============================================================================
    simulator_path = os.path.join(data_dir, domain_name, abstraction_name, simulation_name)
    if os.path.isfile(simulator_path + ".yaml"):
        print(colored("Simulation was already done.", "blue"))
        if not force:
            return

    # Initialize Simulator
    current_ground_state = INITIAL_GROUND_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)
    logging.info("Initialized the current ground state: [%s]", current_ground_state)
    logging.info("Initialized the current abstract state: [%s]", current_abstract_state)
    log["Simulation"] = {"Steps": []}

    state_history = []
    policy_cache = {}

    logging.info("Activating the simulator...")
    time_step = 1
    while time_step <= time_horizon:
        logging.info(f"Time step: {time_step}")
        log["Simulation"]["Steps"].append({})
        iteration_log = log["Simulation"]["Steps"][-1]

        time_step += 1

        iteration_log["Step"] = time_step
        iteration_log["Current Ground State"] = current_ground_state

        ground_states = abstract_mdp.get_ground_states([current_abstract_state])

        if current_ground_state not in policy_cache:
            logging.info("Encountered a new abstract state: [%s]", current_abstract_state)

            logging.info("Starting the policy sketch refine algorithm...")
            start = time.time()
            solution = policy_sketch_refine.solve(ground_mdp, current_ground_state, abstract_mdp,
                                                  current_abstract_state, expand_poi, gamma)
            end = time.time()
            logging.info("Finished the policy sketch refine algorithm: [time=%f]", end - start)
            iteration_log["Policy-Sketch-Refine Time"] = end - start
            iteration_log["Policy-Sketch-Refine Human Time"] = readable_time(end - start)

            start = time.time()
            values = utils.get_ground_entities(solution['values'], ground_mdp, abstract_mdp)
            logging.info("Calculated the values from the solution of policy sketch refine: [time=%f]",
                         time.time() - start)

            start = time.time()
            policy = utils.get_ground_policy(values, ground_mdp, abstract_mdp, ground_states,
                                             current_abstract_state, gamma)
            logging.info("Calculated the policy from the values: [time=%f]", time.time() - start)

            logging.info("Cached the ground states for the new abstract state: [%s]", current_abstract_state)
            for ground_state in ground_states:
                policy_cache[ground_state] = policy[ground_state]

        state_history.append(current_ground_state)

        expanded_state_policy = {}
        for ground_state in ground_states:
            expanded_state_policy[ground_state] = policy_cache[ground_state]

        current_action = policy_cache[current_ground_state]

        logging.info("Current Ground State: [%s]", current_ground_state)
        logging.info("Current Abstract State: [%s]", current_abstract_state)
        logging.info("Current Action: [%s]", current_action)

        printer.print_earth_observation_policy(ground_mdp,
                                               state_history=state_history,
                                               expanded_state_policy=expanded_state_policy,
                                               policy_cache=policy_cache)

        current_ground_state = utils.get_successor_state(current_ground_state, current_action, ground_mdp)
        current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)

        if sleep_duration > 0:
            time.sleep(sleep_duration)

    log["Simulation"]["Number of Steps"] = time_step - 1

    yaml.dump(log, open(simulator_path + ".yaml", "w"))


if __name__ == '__main__':
    main()
