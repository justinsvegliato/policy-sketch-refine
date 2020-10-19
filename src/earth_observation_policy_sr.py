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


def get_domain_path(data_dir, config):
    domain_name = f"Earth_Observation_W{config['width']}_H{config['height']}_I{config['n_pois']}_" \
                  f"V{config['visibility']}_v{config['domain_variation']}"

    domain_name = os.path.join(data_dir, domain_name)

    if not os.path.isdir(domain_name):
        os.mkdir(domain_name)

    return domain_name


def get_abstraction_path(data_dir, config):
    abstraction_name = f"W{config['abstract_width']}_H{config['abstract_height']}"

    abstraction_name = os.path.join(get_domain_path(data_dir, config), abstraction_name)

    if not os.path.isdir(abstraction_name):
        os.mkdir(abstraction_name)

    return abstraction_name


def get_simulator_path(data_dir, config):
    simulation_name = f"Simulation_s{config['sleep_duration']}_T{config['time_horizon']}_" \
                      f"gamma{config['gamma']}_Expand{config['expand_poi']}_v{config['simulation_variation']}"

    simulator_path = os.path.join(get_abstraction_path(data_dir, config), simulation_name)

    if not os.path.isdir(simulator_path):
        os.mkdir(simulator_path)

    return simulator_path


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
    arg_parser.add_argument("-vs", "--sim-variation", required=True, type=int)
    arg_parser.add_argument("-f", "--force", default="")

    # =========================================================================================
    # Read args
    # =========================================================================================
    args = arg_parser.parse_args()
    domain_variation = args.random_variation
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
    simulation_variation = args.simulation_variation
    expand_poi = args.expand_poi.lower() in ("1", "yes", "y", "t", "true")
    force = args.force.lower() in ("1", "yes", "y", "t", "true")

    # =========================================================================================
    # Run
    # =========================================================================================
    config = {
        "domain_variation": domain_variation,
        "width": width,
        "height": height,
        "n_pois": points_of_interest,
        "visibility": visibility,
        "abstract_aggregate": abstract_aggregate,
        "abstract_width": abstract_width,
        "abstract_height": abstract_height,
        "simulation_variation": simulation_variation,
        "sleep_duration": sleep_duration,
        "time_horizon": time_horizon,
        "gamma": gamma,
        "expand_poi": expand_poi,
    }

    run(data_dir, config, simulate=True, force=force)

def construct_abstract_mdp(ground_mdp, abstract_mdp_file_path, config):

    # Abstract MDP
    if os.path.isfile(abstract_mdp_file_path + ".pickle") and os.path.isfile(abstract_mdp_file_path + ".yaml"):
        print(colored("Abstraction was already done.", "blue"))
        #abstract_mdp = pickle.load(open(abstract_mdp_file_path + ".pickle", "rb"))
        #log = yaml.load(open(abstract_mdp_file_path + ".yaml"), Loader=yaml.FullLoader)
    else:
        start = time.time()
        abstract_mdp = EarthObservationAbstractMDP(
            ground_mdp, config["abstract_aggregate"], config["abstract_width"], config["abstract_height"])
        end = time.time()

        logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]",
                     len(abstract_mdp.states()),
                     len(abstract_mdp.actions()),
                     end - start)

        # Store abstract MDP
        with open(abstract_mdp_file_path + ".pickle", "wb") as f:
            print(pickle.dump(abstract_mdp, f, pickle.HIGHEST_PROTOCOL))

        # Store abstraction logs
        log = {}
        log["Abstract MDP"] = {
            "Abstraction Aggregate": config["abstract_aggregate"],
            "Abstraction Width": config["abstract_width"],
            "Abstraction Height": config["abstract_height"],
            "Abstraction Time": round(end - start, 2),
            "Abstraction Human Time": readable_time(end - start),
            "Abstraction Number of States": len(abstract_mdp.states()),
        }
        yaml.dump(log, open(abstract_mdp_file_path + ".yaml", "w"))

def simulate_MDP(log, ground_mdp, data_dir, config, force, solution, abstract_mdp):

    # ==============================================================================
    # MDP Simulator
    # ==============================================================================
    simulator_path = get_simulator_path(data_dir, config)
    if os.path.isfile(simulator_path + ".yaml"):
        print(colored("Simulation was already done.", "blue"))
        if not force:
            return

    # Initialize Simulator
    current_ground_state = INITIAL_GROUND_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)
    logging.info("Initialized the current ground state: [%s]", current_ground_state)
    logging.info("Initialized the current abstract state: [%s]", current_abstract_state)
    log["Simulation"] = {
        "Variation": config["simulation_variation"],
        "Cumulative Reward": 0,
        "Steps": []
    }

    start = time.time()
    values = utils.get_ground_entities(solution['values'], ground_mdp, abstract_mdp)
    end = time.time()
    logging.info("Calculated the values for the ground MDP: [time=%f]", end - start)
    log["Ground Entities Time"] = end - start
    log["Ground Entities Human Time"] = readable_time(end - start)

    start = time.time()
    policy = utils.get_full_ground_policy(values, ground_mdp, ground_mdp.states(), config["gamma"])
    end = time.time()
    logging.info("Calculated the policy from the values: [time=%f]", end - start)
    log["Ground Policy Time"] = end - start
    log["Ground Policy Human Time"] = readable_time(end - start)

    logging.info("Activating the simulator...")
    time_step = 1
    utils.set_simulation_random_variation(config["simulation_variation"])
    while time_step <= config["time_horizon"]:
        logging.info(f"Time step: {time_step}")
        log["Simulation"]["Steps"].append({})
        step_log = log["Simulation"]["Steps"][-1]

        time_step += 1

        step_log["Step"] = time_step
        step_log["Current Ground State"] = current_ground_state
        step_log["Current Abstract State"] = current_abstract_state

        current_action = policy[current_ground_state]

        logging.info("Current Ground State: [%s]", current_ground_state)
        logging.info("Current Abstract State: [%s]", current_abstract_state)
        logging.info("Current Action: [%s]", current_action)

        current_reward = ground_mdp.reward_function(current_ground_state, current_action)
        step_log["Current Reward"] = current_reward
        log["Simulation"]["Cumulative Reward"] += current_reward

        current_ground_state = utils.get_successor_state(current_ground_state, current_action, ground_mdp)
        current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)

        if config["sleep_duration"] > 0:
            time.sleep(config["sleep_duration"])

    log["Simulation"]["Number of Steps"] = time_step - 1

    yaml.dump(log, open(simulator_path + ".yaml", "w"))

def simulate_PAMDP(log, ground_mdp, abstract_mdp, data_dir, config, force):

    # ==============================================================================
    # PAMDP Simulator
    # ==============================================================================
    simulator_path = get_simulator_path(data_dir, config)
    if os.path.isfile(simulator_path + ".yaml"):
        print(colored("Simulation was already done.", "blue"))
        if not force:
            return

    # Initialize Simulator
    current_ground_state = INITIAL_GROUND_STATE
    current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)
    logging.info("Initialized the current ground state: [%s]", current_ground_state)
    logging.info("Initialized the current abstract state: [%s]", current_abstract_state)
    log["Simulation"] = {
        "Variation": config["simulation_variation"],
        "Cache Misses": 0,
        "Cache Hits": 0,
        "Cumulative Reward": 0,
        "Steps": []
    }

    #state_history = []
    policy_cache = {}

    logging.info("Activating the simulator...")
    time_step = 1
    utils.set_simulation_random_variation(config["simulation_variation"])
    while time_step <= config["time_horizon"]:
        logging.info(f"Time step: {time_step}")
        log["Simulation"]["Steps"].append({})
        step_log = log["Simulation"]["Steps"][-1]

        time_step += 1

        step_log["Step"] = time_step
        step_log["Current Ground State"] = current_ground_state
        step_log["Current Abstract State"] = current_abstract_state

        ground_states = abstract_mdp.get_ground_states([current_abstract_state])

        if current_ground_state not in policy_cache:
            logging.info("Encountered a new abstract state: [%s]", current_abstract_state)
            log["Simulation"]["Cache Misses"] += 1

            logging.info("Starting the policy sketch refine algorithm...")
            start = time.time()
            solution = policy_sketch_refine.solve(ground_mdp, current_ground_state, abstract_mdp,
                                                  current_abstract_state, config["expand_poi"], config["gamma"])
            end = time.time()
            logging.info("Finished the policy sketch refine algorithm: [time=%f]", end - start)
            step_log["Policy Sketch-Refine Time"] = end - start
            step_log["Policy Sketch-Refine Human Time"] = readable_time(end - start)

            start = time.time()
            values = utils.get_ground_entities(solution['values'], ground_mdp, abstract_mdp)
            end = time.time()
            logging.info("Calculated the values from the solution of policy sketch refine: [time=%f]", end - start)
            step_log["Ground Entities Time"] = end - start
            step_log["Ground Entities Human Time"] = readable_time(end - start)

            start = time.time()
            policy = utils.get_ground_policy(values, ground_mdp, abstract_mdp, ground_states,
                                             current_abstract_state, config["gamma"])
            end = time.time()
            logging.info("Calculated the policy from the values: [time=%f]", end - start)
            step_log["Ground Policy Time"] = end - start
            step_log["Ground Policy Human Time"] = readable_time(end - start)

            logging.info("Cached the ground states for the new abstract state: [%s]", current_abstract_state)
            for ground_state in ground_states:
                policy_cache[ground_state] = policy[ground_state]
        else:
            log["Simulation"]["Cache Hits"] += 1

        #state_history.append(current_ground_state)

        #expanded_state_policy = {}
        #for ground_state in ground_states:
        #    expanded_state_policy[ground_state] = policy_cache[ground_state]

        current_action = policy_cache[current_ground_state]

        logging.info("Current Ground State: [%s]", current_ground_state)
        logging.info("Current Abstract State: [%s]", current_abstract_state)
        logging.info("Current Action: [%s]", current_action)

        current_reward = ground_mdp.reward_function(current_ground_state, current_action)
        step_log["Current Reward"] = current_reward
        log["Simulation"]["Cumulative Reward"] += current_reward

        current_ground_state = utils.get_successor_state(current_ground_state, current_action, ground_mdp)
        current_abstract_state = abstract_mdp.get_abstract_state(current_ground_state)

        if config["sleep_duration"] > 0:
            time.sleep(config["sleep_duration"])

    log["Simulation"]["Number of Steps"] = time_step - 1
    log["Simulation"]["Cache Hit Ratio"] = log["Simulation"]["Cache Hits"] / log["Simulation"]["Number of Steps"]
    log["Simulation"]["Cache Miss Ratio"] = log["Simulation"]["Cache Misses"] / log["Simulation"]["Number of Steps"]

    yaml.dump(log, open(simulator_path + ".yaml", "w"))


def run(data_dir, config, simulate=False, force=False):
    size = config["width"], config["height"]

    # Check data dir
    if not os.path.isdir(data_dir):
        raise Exception(f"Data directory {data_dir} does not exist. Create it and run this again.")

    # Generate Earth Observation MDP
    utils.set_domain_random_variation(config["domain_variation"])
    start = time.time()
    ground_mdp = EarthObservationMDP(size, config["n_pois"], config["visibility"])
    end = time.time()
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]",
                 len(ground_mdp.states()),
                 len(ground_mdp.actions()),
                 end - start)

    abstract_mdp_file_path = get_abstraction_path(data_dir, config)

    if not simulate:
        construct_abstract_mdp(ground_mdp, abstract_mdp_file_path, config)
    else:
        print(abstract_mdp_file_path)
        if os.path.isfile(abstract_mdp_file_path + ".pickle") and os.path.isfile(abstract_mdp_file_path + ".yaml"):
            print(colored("Loading abstract MDP from cache.", "blue"))
            # Load the abstract MDP 
            abstract_mdp = pickle.load(open(abstract_mdp_file_path + ".pickle", "rb"))
            # Simulate the PAMDP
            if config["abstract_aggregate"] == "MEAN":
                log = yaml.load(open(abstract_mdp_file_path + ".yaml"), Loader=yaml.FullLoader)
                simulate_PAMDP(log, ground_mdp, abstract_mdp, data_dir, config, force)
            # Solve and simulate the ground MDP only (no abstraction)
            elif config["abstract_aggregate"] == "NONE":
                log = {
                    "Earth Observation Ground MDP": {
                        "Variation": config["domain_variation"],
                        "Width": config["width"],
                        "Height": config["height"],
                        "POIs": config["n_pois"],
                        "Visibility": config["visibility"],
                        "Random Variation": config["domain_variation"],
                        "Number of States": len(ground_mdp.states()),
                        "Number of Actions": len(ground_mdp.actions()),
                        "Creation Time": round(end - start, 2),
                        "Creation Human Time": readable_time(end - start),
                    }
                }
                print(colored("Solving ground MDP.", "blue"))
                start = time.time()
                solution = cplex_mdp_solver.solve(ground_mdp, config["gamma"])
                end = time.time()
                log["Earth Observation Ground MDP"]["Solving Time"] = round(end - start, 2)
                log["Earth Observation Ground MDP"]["Solving Human Time"] = readable_time(end - start)
                simulate_MDP(log, ground_mdp, data_dir, config, force, solution, abstract_mdp)
            else:
                raise AssertionError("Abstract_aggregate not recognized")
        else:
            raise AssertionError("No abstraction found in data folder but you wanted to simulate now.")
            return

        #printer.print_earth_observation_policy(ground_mdp,
        #                                       state_history=state_history,
        #                                       expanded_state_policy=expanded_state_policy,
        #                                       policy_cache=policy_cache)


if __name__ == '__main__':
    main()
