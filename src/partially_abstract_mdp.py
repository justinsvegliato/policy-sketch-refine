import math
import statistics

class AbstractMDP:
    def __compute_abstract_states(self, ground_mdp):
        return None

    def __compute_abstract_rewards(self, ground_mdp):
        return None

    def __compute_abstract_transition_probabilities(self, ground_mdp, abstract_mdp):
        return None

    def __compute_abstract_start_state_probabilities(self, ground_mdp):
        return None

    def __init__(self, ground_mdp, abstract_mdp):
        self.abstract_states = self.__compute_abstract_states(ground_mdp)
        self.abstract_actions = ground_mdp.actions()
        self.abstract_rewards = self.__compute_abstract_rewards(ground_mdp)
        self.abstract_transition_probabilities = self.__compute_abstract_transition_probabilities(ground_mdp, abstract_mdp)
        self.abstract_start_state_probabilities = self.__compute_abstract_start_state_probabilities(ground_mdp)

    def states(self):
        return list(self.abstract_states)

    def actions(self):
        return self.abstract_actions

    def transition_function(self, state, action, successor_state):
        return self.abstract_transition_probabilities[state][action][successor_state]

    def reward_function(self, state, action):
        return self.abstract_rewards[state][action]

    def start_state_function(self, state):
        return self.abstract_start_state_probabilities[state]

    def get_abstract_state(self, ground_state):
        for abstract_state, ground_states in self.abstract_states.items():
            if ground_state in ground_states:
                return abstract_state
        return None

    def get_ground_states(self, abstract_state):
        return self.abstract_states[abstract_state]
