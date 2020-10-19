from argparse import ArgumentParser
from run import get_x_y
import matplotlib.pyplot as plt


# Use a function for the X axis
def size(config, results):
    # In this example, I'm setting the total size in the X axis
    return config["width"] * config["height"]


# Another function for the X axis
def n_states(config, results):
    # In this example, I'm using the number of states
    return results["Earth Observation Ground MDP"]["Number of States"]

# 
def reward_density(config, results):
    # Number of POIs divided by number of locations (all weather configs have reward at poi losc)
    return float(results["Earth Observation Ground MDP"]["POIs"]) / float(results["Earth Observation Ground MDP"]["Width"] * results["Earth Observation Ground MDP"]["Height"])

# Use a function for the Y axis
def cumulative_sketch_refine_time(config, results):
    return sum(s.get("Policy Sketch-Refine Time", 0) for s in results["Simulation"]["Steps"])


# Use a function for the Y axis
def ground_mdp_solve_time(config, results):
    return results["Earth Observation Ground MDP"]["Solving Time"]


# Another function for the Y axis
def cumulative_reward(config, results):
    return results["Simulation"]["Cumulative Reward"]

# Use a function for the Y axis
def percent_states_expanded(config, results):
    num_abstract_states = results["Abstract MDP"]["Abstraction Number of States"]
    list_abstract_states = [s.get("Current Abstract State", 0) for s in results["Simulation"]["Steps"]]
    set_abstract_states = set(list_abstract_states)
    return float(len(set_abstract_states)) / float(num_abstract_states)

#TODO: add code to support different expansion strategies
#TODO: increase horizon
#TODO: add more trials
#          add code to create error bars over trials
#TODO: determine which problems to run


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("baseline_config_file")
    arg_parser.add_argument("config_file")
    arg_parser.add_argument("data_dir")
    
    args = arg_parser.parse_args()
    baseline_config_file = args.baseline_config_file
    config_file = args.config_file
    data_dir = args.data_dir



    # time vs. size

    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=size, y_func=ground_mdp_solve_time, sort=True)
    x, y = get_x_y(data_dir, config_file, x_func=size, y_func=cumulative_sketch_refine_time, sort=True)

    plt.plot(b_x, b_y, 'r', x, y, 'b')
    plt.show()
    


    #: cum reward ratio vs. size

    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=size, y_func=cumulative_reward, sort=True)
    x, y = get_x_y(data_dir, config_file, x_func=size, y_func=cumulative_reward, sort=True)

    # NOTE: sanity check for correctness
    y = [40, 35]

    y = [i / j for i, j in zip(y, b_y)]
    b_y = [1.0 for _ in range(len(y))]

    plt.plot(x, b_y, 'r--', x, y, 'b')
    plt.show()



    # cum reward ratio vs. reward density
    
    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x, y = get_x_y(data_dir, config_file, x_func=reward_density, y_func=cumulative_reward, sort=True)

    # NOTE: sanity check for correctness
    y = [40, 35]

    y = [i / j for i, j in zip(y, b_y)]
    b_y = [1.0 for _ in range(len(y))]

    plt.plot(x, b_y, 'r--', x, y, 'b')
    plt.show()



    # percentage of states expanded vs. reward density
    
    x, y = get_x_y(data_dir, config_file, x_func=reward_density, y_func=percent_states_expanded, sort=True)
    plt.plot(x, y, 'b')
    plt.show()


if __name__ == '__main__':
    main()
