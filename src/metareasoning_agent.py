import copy
import random
from collections import deque

import torch
from torch import nn


class MetareasoningAgent:

    def __init__(self, seed, layer_sizes, learning_rate, sync_frequency, experience_buffer_size):
        torch.manual_seed(seed)

        self.action_value_network = self.build_neural_network(layer_sizes)
        self.target_action_value_network = copy.deepcopy(self.action_value_network)

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.action_value_network.parameters(), lr=learning_rate)

        self.sync_frequency = sync_frequency
        self.sync_counter = 0

        self.gamma = torch.tensor(0.95).float()
        self.experience_buffer = deque(maxlen=experience_buffer_size)

    # TODO Replace the neural network with a class for readability/adjustability
    def build_neural_network(self, layer_sizes):
        assert len(layer_sizes) > 1

        layers = []
        for index in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            activation_layer = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear_layer, activation_layer)

        return nn.Sequential(*layers)

    def load_model(self, model_path):
        self.action_value_network.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.action_value_network.state_dict(), model_path)

    def get_action(self, state, action_space_size, epsilon):
        with torch.no_grad():
            action_values = self.action_value_network(torch.from_numpy(state).float())

        _, best_actions = torch.max(action_values, axis=0)
        best_actions = best_actions if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_size, (1,))

        return best_actions

    def get_next_action_value(self, state):
        with torch.no_grad():
            action_values = self.target_action_value_network(state)

        best_action_values, _ = torch.max(action_values, axis=1)

        return best_action_values

    def collect_experience(self, experience):
        self.experience_buffer.append(experience)

    def sample_experience(self, batch_size):
        if len(self.experience_buffer) < batch_size:
            batch_size = len(self.experience_buffer)

        batch = random.sample(self.experience_buffer, batch_size)

        states = torch.tensor([experience[0] for experience in batch]).float()
        actions = torch.tensor([experience[1] for experience in batch]).float()
        rewards = torch.tensor([experience[2] for experience in batch]).float()
        next_states = torch.tensor([experience[3] for experience in batch]).float()

        return states, actions, rewards, next_states

    def train(self, batch_size):
        states, _, rewards, next_states = self.sample_experience(batch_size)

        if self.sync_counter == self.sync_frequency:
            self.target_action_value_network.load_state_dict(self.action_value_network.state_dict())
            self.sync_counter = 0

        action_values = self.action_value_network(states)
        best_action_values, _ = torch.max(action_values, axis=1)

        next_action_value_batch = self.get_next_action_value(next_states)
        target_return = rewards + self.gamma * next_action_value_batch

        loss = self.loss_function(best_action_values, target_return)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.sync_counter += 1

        return loss.item()
