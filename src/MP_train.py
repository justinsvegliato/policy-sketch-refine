import sys
from typing import DefaultDict
import yaml
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from earth_observation_policy_sr import run
from MP_run import read_config
from MP_LEARNING import Agents
from MP_plot import plot_all
import matplotlib.pyplot as plt
import seaborn as sns
import json


def csv_config_name(data_folder, expansion_strat):
    fn = Path(data_folder) / f'config_{expansion_strat}.csv'
    return fn


def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_folders(folder_paths):
    for fp in folder_paths:
        Path(fp).mkdir(parents=True, exist_ok=True)

def gen_csvs(gc):
    expansion_strats = gc['expansion_strategies']
    GEOD = gc['GEOD']
    AEOD = gc['AEOD']
    SIM = gc['SIMULATION']

    cols = [
        'width',
        'height',
        'n_pois',
        'visibility',
        'expansion_level',
        'sleep_duration',
        'time_horizon',
        'abstract_aggregate',
        'abstract_width',
        'abstract_height',
        'gamma',
        'expand_poi',
        'domain_variation',
        'simulation_variation'
    ]
    rows = []

    GEOD_widths = list(range(GEOD['width']['min'], GEOD['width']['max']))
    GEOD_heights = list(range(GEOD['height']['min'], GEOD['height']['max']))
    GEOD_pois = list(range(GEOD['n_points_of_interest']['min'], GEOD['n_points_of_interest']['max']))

    AEOD_widths = list(range(AEOD['width']['min'], AEOD['width']['max']))
    AEOD_heights = list(range(AEOD['height']['min'], AEOD['height']['max']))

    sim_variations = list(range(SIM['sim_variation']['min'], SIM['sim_variation']['max']))
    domain_variations = list(range(SIM['domain_variation']['min'], SIM['domain_variation']['max']))

    ranges = [GEOD_widths, GEOD_heights, GEOD_pois, AEOD_widths, AEOD_heights, sim_variations, domain_variations]
    range_combos = list(itertools.product(*ranges))

    for combo in range_combos:
        GEOD_width, GEOD_height, GEOD_poi, AEOD_width, AEOD_height, sim_variation, domain_variation = combo
        rows.append(
            (
                GEOD_width, 
                GEOD_height, 
                GEOD_poi, 
                GEOD['visibility'],
                '-1',
                SIM['sleep_duration'],
                SIM['time_horizon'],
                AEOD['aggregation'],
                AEOD_width, 
                AEOD_height,
                SIM['gamma'],
                SIM['expand_poi'],
                domain_variation,
                sim_variation
            )
        )
    # print(rows)
    df = pd.DataFrame(rows, columns=cols)

    for expansion_strat in expansion_strats:
        fn = csv_config_name(gc['data_folder'], expansion_strat)

        # replace value with a few extra rules
        exp_strat_val = expansion_strat
        if expansion_strat == 'ground':
            exp_strat_val = '0'
        elif expansion_strat == 'abstract':
            exp_strat_val = 'a'

        df['expansion_level'] = exp_strat_val
        
        # Write CSV
        df.to_csv(fn, index=False)



def generate_data(gc):
    expansion_config_paths = [csv_config_name(gc['data_folder'], expansion_strat) for expansion_strat in gc['expansion_strategies']]
    expansion_configs = [read_config(config_path) for config_path in expansion_config_paths]
    # Generate the initial abstraction
    for _, config in expansion_configs[0].iterrows():
        print(config)
        run(gc['data_folder'], config, simulate=False, force=False)

    # Simulate on all
    for expansion_config in expansion_configs:
        for _, config in expansion_config.iterrows():
            run(gc['data_folder'], config, simulate=True, force=False)



def generate_plots(gc):
    # config_files = [csv_config_name(gc['data_folder'], expansion_strat) for expansion_strat in gc['expansion_strategies']]
    # plot_all(config_files, gc['data_folder'])

    # Read in all the log data
    data = {}
    for EOD_config in Path(gc['data_folder']).iterdir():
        if EOD_config.is_dir():
            data[EOD_config] = DefaultDict(list)
            for nest in EOD_config.iterdir():
                if nest.is_dir():
                    for run in nest.iterdir():
                        if run.suffix == '.json':
                            method = str(run).split('_')[-5]
                            data[EOD_config][method].append(run)
                        # print(run)

    graph_data = {}
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    for EOD_config, expansion_strats in data.items():
        color_count = 0
        for expansion_strat, runs in expansion_strats.items(): 
            # data[EOD_config][expansion_strat]['results'] = []
            for run in runs:
                with open(run, 'r') as f:
                    print(run)
                    run_log = json.load(f)
                    # print(run_log)
                    # Analyze simulation
                    time = 0
                    reward = 0

                    times = []
                    rewards = []
                    # print(run_log)
                    for step in run_log['Simulation']['Steps']:
                        if 'Policy Sketch-Refine Time' in step:
                            time += step['Policy Sketch-Refine Time']
                        reward += step['Current Reward']
                        times.append(time)
                        rewards.append(reward)
                    df = pd.DataFrame({'Time': times, 'Reward': rewards})
                    sns.lineplot(x='Time', y='Reward', data=df, color=colors[color_count])
            color_count += 1
        plt.title(str(EOD_config))
        # ax.legent
        plt.savefig(Path(gc['data_folder'])/f'{str(EOD_config).split("/")[-1]}.png')
        plt.clf()



    # # sns.color_palette("tab10")
    # sns.color_palette("hls", 8)
    # graph_data = {}
    # colors = ['r', 'g', 'b', 'c', 'm', 'k']
    # for EOD_config, expansion_strats in data.items():
    #     color_count = 0
    #     for expansion_strat, runs in expansion_strats.items(): 
    #         # data[EOD_config][expansion_strat]['results'] = []
    #         for run in runs:
    #             with open(run, 'r') as f:
    #                 print(run)
    #                 run_log = json.load(f)
    #                 # print(run_log)
    #                 # Analyze simulation
    #                 time = 0
    #                 reward = 0

    #                 times = []
    #                 rewards = []
    #                 # print(run_log)
    #                 for step in run_log['Simulation']['Steps']:
    #                     if 'Policy Sketch-Refine Time' in step:
    #                         time += step['Policy Sketch-Refine Time']
    #                     reward += step['Current Reward']
    #                     times.append(time)
    #                     rewards.append(reward)
    #                 df = pd.DataFrame({'Time': times, 'Reward': rewards, 'Strat': color_count})
    #                 # sns.lineplot(x='Time', y='Reward', data=df, color=colors[color_count])
    #                 sns.lineplot(x='Time', y='Reward', data=df, hue='Strat')
    #         color_count += 1
    #     plt.title(str(EOD_config))
    #     # ax.legent
    #     plt.savefig(Path(gc['data_folder'])/f'{str(EOD_config).split("/")[-1]}.png')
    #     plt.clf()




    # Generate total time graph
    pass


    

def agent():
    pass

def train():
    pass

def eval():
    pass


def main(args):
    global_config = parse_yaml(args.yaml)
    gc = global_config
    create_folders([gc['data_folder'], gc['rl_hyper']['model_folder']])

    # Generate config CSV
    gen_csvs(gc)

    # Generate all data
    generate_data(gc)


    # Generate plots
    generate_plots(gc)



    




def parse_args():
    arg_parser = ArgumentParser()

    # Data Folder (use a partition with enough space)
    arg_parser.add_argument("-y", "--yaml", required=True)

    # # Earth Observation
    # arg_parser.add_argument("-W", "--width", required=True, type=int)
    # arg_parser.add_argument("-H", "--height", required=True, type=int)
    # arg_parser.add_argument("-I", "--n-points-of-interest", required=True, type=int)
    # arg_parser.add_argument("-V", "--visibility")
    # arg_parser.add_argument("-v", "--random-variation", required=True, type=int)

    # # Abstraction
    # arg_parser.add_argument("-aA", "--abstract-aggregate", required=True)
    # arg_parser.add_argument("-aW", "--abstract-width", required=True, type=int)
    # arg_parser.add_argument("-aH", "--abstract-height", required=True, type=int)


    # =========================================================================================
    # Read args
    # =========================================================================================
    args = arg_parser.parse_args()

    # Validate args
    if not Path(args.yaml).exists():
        print(f'Yaml config file does not exist: {args.yaml}')
        sys.exit()


    return args


if __name__ == '__main__':
    main(parse_args())