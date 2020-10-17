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


# Use a function for the Y axis
def cumulative_sketch_refine_time(config, results):
    return sum(s.get("Policy Sketch-Refine Time", 0) for s in results["Simulation"]["Steps"])


# Another function for the Y axis
def cumulative_reward(config, results):
    return results["Simulation"]["Cumulative Reward"]


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("config_file")
    arg_parser.add_argument("data_dir")

    args = arg_parser.parse_args()
    config_file = args.config_file
    data_dir = args.data_dir

    # x, y = get_x_y(data_dir, config_file, x_func=size, y_func=cumulative_sketch_refine_time, sort=True)
    # x, y = get_x_y(data_dir, config_file, x_func=size, y_func=cumulative_reward, sort=True)
    x, y = get_x_y(data_dir, config_file, x_func=n_states, y_func=cumulative_reward, sort=True)

    print(x)
    print(y)

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
