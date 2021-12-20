import logging 
import json
import pickle

from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from MP_Features import Features
from earth_observation_mdp import EarthObservationMDP
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_policy_sr import construct_abstract_mdp, get_abstraction_path, simulate_MDP, simulate_PAMDP, get_simulator_path


DATA_DIR = 'MP_TORCH'
TOTAL_FEATURES = 28 # 16 state features + 14 config features
TOTAL_CHOICES = 4
config = {
    'width': 9,
    'height': 9,
    'n_pois': 2,
    'visibility': 1,
    'abstract_aggregate': 'MEAN',
    'abstract_width': 3,
    'abstract_height': 3,
    'domain_variation': 1,
    'simulation_variation': 1,
    'time_horizon': 2000,
    'expansion_level': 'ML',
    'gamma': .99,
    'expand_poi': True,
    'sleep_duration': 1e-5
}


class MLP(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(D_in, H) 
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred


logging.basicConfig(
    format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger()
logger.disabled = True

def calc_max_reward(mdp, time_horizon):
    states = mdp.states()
    actions = mdp.actions()

    max_reward = float('-inf')
    for state in states:
        for action in actions:
            move_reward = mdp.reward_function(state, action)
            max_reward = max(max_reward, move_reward)

    return max_reward * time_horizon
        


def loss_func(output, max_reward):
    reward_diff = max_reward - output
    return reward_diff

def train():
    model = MLP(TOTAL_FEATURES, 16, TOTAL_CHOICES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create training dataset
    ground_mdp = EarthObservationMDP(
        (config['width'], config['height']), 
        config['n_pois'], 
        config['visibility']
    )
    max_reward = calc_max_reward(ground_mdp, config['time_horizon'])
    print(f'Max reward: {max_reward}')
    abstract_mdp_file_path = get_abstraction_path(DATA_DIR, config)
    construct_abstract_mdp(ground_mdp, abstract_mdp_file_path, config)
    log = json.load(open(f'{abstract_mdp_file_path}.json'))
    abstract_mdp = pickle.load(open(abstract_mdp_file_path + ".pickle", "rb"))
    eod_configs = [abstract_mdp]

    # Training loop
    for epoch in range(15):
        print(f'Training epoch: {epoch}')
        curr_loss = 0.0

        # Iterate over EOD configs
        for eod_config in eod_configs:
            # features = deepcopy(config).update()
            features = torch.rand(TOTAL_FEATURES)
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(features)
            # Simulate the MDP in forward pass
            ML_params = {
                'model': model,
                'expansion_choices': [2, 'a', 1, 0],
                'config': config,
                'training_mode': True
            }
            simulate_PAMDP(
                log=log, 
                ground_mdp=ground_mdp, 
                data_dir=DATA_DIR,
                abstract_mdp=abstract_mdp, 
                config=config, 
                force=False, 
                ML_params=ML_params
            )
            # Read back the reward
            sim_path = get_simulator_path(DATA_DIR, config)
            sim_results = json.load(open(f'{sim_path}.json'))
            # print(sim_results)
            cumulative_reward = sim_results['Simulation']['Cumulative Reward']
            # print(cumulative_reward)

            # Loss calculation

            loss = loss_func(cumulative_reward, max_reward)

            # Optimize
            optimizer.step()

            # Current Progress (later add in tensorboard)
            curr_loss += loss
            print(f'Curr loss: {curr_loss}')
    print(f'Training Complete')




        


def main():
    # create ground MDP
    ground_mdp = EarthObservationMDP(
        (config['width'], config['height']), 
        config['n_pois'], 
        config['visibility']
    )
    # Create abstracted MDP
    abstract_mdp_file_path = get_abstraction_path(DATA_DIR, config)
    construct_abstract_mdp(ground_mdp, abstract_mdp_file_path, config)

    # Load back
    log = json.load(open(f'{abstract_mdp_file_path}.json'))
    abstract_mdp = pickle.load(open(abstract_mdp_file_path + ".pickle", "rb"))

    # Calculate global features
    features = Features(ground_mdp)


    # Train 
    train()



    # # Calculate features for a specific state
    # ground_state = ground_mdp.states()[1]
    # abstract_state = abstract_mdp.get_abstract_state(ground_state)
    # state_0_features = features.state_features(ground_state, abstract_mdp, abstract_state, config['gamma'])
    # print(f'Feats: {state_0_features}')
    # state_0_flat = features.flatten_state_features(state_0_features)
    # print(f'Flat Feats: {state_0_flat}')
    # print(f'Flat Feats: {len(state_0_flat)}')


    # # Test getting reward
    # # Simulate the MDP
    # ML_params = {
    #     'model': None,
    #     'expansion_choices': [2, 'a', 1, 0],
    #     'training_mode': True
    # }
    # simulate_PAMDP(
    #     log=log, 
    #     ground_mdp=ground_mdp, 
    #     data_dir=DATA_DIR,
    #     abstract_mdp=abstract_mdp, 
    #     config=config, 
    #     force=False, 
    #     ML_params=ML_params
    # )

    # # Read back the reward
    # sim_path = get_simulator_path(DATA_DIR, config)
    # sim_results = json.load(open(f'{sim_path}.json'))
    # # print(sim_results)
    # cumulative_reward = sim_results['Simulation']['Cumulative Reward']
    # print(cumulative_reward)



    # num_random = 10
    # num_states = len(abstract_mdp.states())
    # random_states = np.random.randint(0, num_states, num_random)
    # for rs in random_states:
    #     curr_abstract_state = 

        

        























if __name__ == '__main__':
    main()