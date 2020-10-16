import pandas as pd
from argparse import ArgumentParser

from earth_observation_policy_sr import run


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("config_file")
    arg_parser.add_argument("data_dir")
    arg_parser.add_argument("action")

    args = arg_parser.parse_args()
    config_file = args.config_file
    data_dir = args.data_dir
    action = args.action

    config = pd.read_csv(config_file)
    print(config)
    print()

    for index, row in config.iterrows():
        print(row)

        width = row["width"]
        height = row["height"]
        points_of_interest = row["n_POIs"]
        visibility = row["visibility"]
        time_horizon = row["time_horizon"]
        random_variation = row["variation"]
        abstract_aggregate = row["a_aggregate"]
        abstract_width = row["a_width"]
        abstract_height = row["a_height"]
        sleep_duration = row["sleep"]
        gamma = row["gamma"]
        expand_poi = row["expand_poi"]

        if action == "abstract":
            simulate = False
        elif action == "simulate":
            simulate = True
        else:
            raise Exception(f"Action {action} not supported")

        run(data_dir,
            random_variation, width, height, points_of_interest, visibility,
            abstract_aggregate, abstract_width, abstract_height,
            sleep_duration, time_horizon,
            gamma, expand_poi,
            simulate=simulate)

        print()


if __name__ == '__main__':
    main()
