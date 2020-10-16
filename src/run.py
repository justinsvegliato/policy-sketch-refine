import pandas as pd
from argparse import ArgumentParser

from earth_observation_policy_sr import run


def main():
    """
    Examples

    1. Set up a data directory on a disk partition large enough to store large datasets.

    2. Set up your CSV configurations files. Look at the example ones.

    3. Run all the experiments in a config file as shown below.

    First, create all the abstractions from the config file:
    $ python src/run.py src/experiments/earth_observation/vary_grid_size/run_config.csv <path-to-data-dir> abstract

    Then, run all the simulations from the same config file:
    $ python src/run.py src/experiments/earth_observation/vary_grid_size/run_config.csv <path-to-data-dir> simulate

    If you want to force simulating everything *again* then add -f=1 to the command line:
    $ python src/run.py src/experiments/earth_observation/vary_grid_size/run_config.csv <path-to-data-dir> simulate -f=1

    ----------

    Best Practice

    * Create different CSV config files, one for each "experiment batch". For example, change the grid size in one,
    and the n of POIs in a different one, etc.

    * Remember to duplicate rows with different domain variations (numbers 1-10) and simulation variations
    (numbers 1-10). Two runs with the same domain and simulation variations will be deterministically the same.
    There will only be a natural small variance in the runtime.
    """

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
        domain_variation = row["variation"]
        abstract_aggregate = row["a_aggregate"]
        abstract_width = row["a_width"]
        abstract_height = row["a_height"]
        sleep_duration = row["sleep"]
        gamma = row["gamma"]
        expand_poi = row["expand_poi"]
        simulation_variation = row["sim_variation"]

        if action == "abstract":
            simulate = False
        elif action == "simulate":
            simulate = True
        else:
            raise Exception(f"Action {action} not supported")

        run(data_dir,
            domain_variation, width, height, points_of_interest, visibility,
            abstract_aggregate, abstract_width, abstract_height,
            simulation_variation, sleep_duration, time_horizon,
            gamma, expand_poi,
            simulate=simulate,
            force=force)

        print()


if __name__ == '__main__':
    main()
