import json

import orjson as orjson
import pandas as pd
import yaml
from argparse import ArgumentParser
from earth_observation_policy_sr import run, run_abstract, get_simulator_path
from multiprocessing import Pool

def read_config(config_file):
    return pd.read_csv(config_file, na_values='null')

def get_simulator_results(data_dir, config):
    simulator_path = get_simulator_path(data_dir, config)

    # Use this for YAML
    # simulator_results = yaml.load(open(simulator_path + ".yaml"), Loader=yaml.CLoader)

    # Use this for JSON
    # try:
    simulator_results = json.load(open(simulator_path + ".json"))
    # except FileNotFoundError:
    #     return None

    return simulator_results


def get_x_y(data_dir, config_file, x_func, y_func, sort=True):
    configs = read_config(config_file)
    # WHile not fully generated
    # x = x[:5]
    # y = y[:5]
    # configs = configs[:55]
    # print(configs)
    # End shortcut
    x = []
    y = []
    for index, config in configs.iterrows():
        #print(config)
        results = get_simulator_results(data_dir, config)
        if results is not None:
            x_val = x_func(config, results)
            y_val = y_func(config, results)
            if y_val == 0:
                y_val = .001
                x.append(x_val)
                y.append(y_val)

    if sort and x and y:
        lists = sorted(zip(*[x, y]))
        x, y = list(zip(*lists))

    return x, y

def run_wrapper(item):
    data_dir, config, simulate, force = item
    run(data_dir, config, simulate=False, force=force)


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

    # items = [(data_dir, config, False, force) for _, config in configs.iterrows()]

    # with Pool() as p:
    #     p.map(run_wrapper, items)
    


    for index, config in configs.iterrows():
        if True:
            # print("am here")
            # run_abstract(data_dir, config, simulate=False, force=force)
            run(data_dir, config, simulate=(action == 'simulate'), force=force)
        else:
            if action == "abstract":
                run(data_dir, config, simulate=False, force=force)
            elif action == "simulate":
                run(data_dir, config, simulate=True, force=force)
            else:
                raise Exception(f"Action {action} not supported")

        # print()


if __name__ == '__main__':
    main()
