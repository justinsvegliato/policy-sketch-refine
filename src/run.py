import pandas as pd
import yaml
from argparse import ArgumentParser
from earth_observation_policy_sr import run, get_simulator_path


def read_config(config_file):
    return pd.read_csv(config_file, na_values='null')

def get_simulator_results(data_dir, config):
    simulator_path = get_simulator_path(data_dir, config)
    simulator_results = yaml.load(open(simulator_path + ".yaml"), Loader=yaml.CLoader)
    return simulator_results


def get_x_y(data_dir, config_file, x_func, y_func, sort=True):
    configs = read_config(config_file)
    x = []
    y = []
    for index, config in configs.iterrows():
        print(config)
        results = get_simulator_results(data_dir, config)
        x.append(x_func(config, results))
        y.append(y_func(config, results))

    if sort:
        lists = sorted(zip(*[x, y]))
        x, y = list(zip(*lists))

    return x, y


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("config_file")
    arg_parser.add_argument("data_dir")
    arg_parser.add_argument("action")
    arg_parser.add_argument("-f", "--force", default="")

    args = arg_parser.parse_args()
    config_file = args.config_file
    data_dir = args.data_dir
    action = args.action
    force = args.force.lower() in ("1", "yes", "y", "t", "true")

    configs = read_config(config_file)
    print(configs)
    print()

    for index, config in configs.iterrows():
        if action == "abstract":
            run(data_dir, config, simulate=False, force=force)
        elif action == "simulate":
            run(data_dir, config, simulate=True, force=force)
        else:
            raise Exception(f"Action {action} not supported")

        print()


if __name__ == '__main__':
    main()
