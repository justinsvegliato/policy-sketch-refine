class PartiallyAbstractMDP:
    # TODO: Move this to another location
    def __compute_weights(self, abstract_mdp):
        weights = {}

        for abstract_state in abstract_mdp.states():
            ground_states = abstract_mdp.get_ground_states(abstract_state)
            for ground_state in ground_states:
                weights[ground_state] = 1 / len(ground_states)

        return weights

    def __compute_abstract_states(self, abstract_mdp, abstract_state):
        ground_states = abstract_mdp.get_ground_states(abstract_state)
        abstract_states = [state for state in abstract_mdp.states() if state != abstract_state]
        return ground_states + abstract_states

    def __compute_abstract_rewards(self, ground_mdp, abstract_mdp):
        abstract_rewards = {}

        for abstract_state in self.abstract_states:
            abstract_rewards[abstract_state] = {}

            for abstract_action in self.abstract_actions:
                if abstract_state in ground_mdp.states():
                    abstract_rewards[abstract_state][abstract_action] = ground_mdp.reward_function(abstract_state, abstract_action)
                else:
                    abstract_rewards[abstract_state][abstract_action] = 0
                    for ground_state in abstract_mdp.get_ground_states(abstract_state):
                        abstract_rewards[abstract_state][abstract_action] += self.weights[ground_state] * ground_mdp.reward_function(ground_state, abstract_action)

        return abstract_rewards

    def __compute_abstract_transition_probabilities(self, ground_mdp, abstract_mdp):
        abstract_transition_probabilities = {}

        for abstract_state in self.abstract_states:
            abstract_transition_probabilities[abstract_state] = {}

            for abstract_action in self.abstract_actions:
                abstract_transition_probabilities[abstract_state][abstract_action] = {}

                for abstract_successor_state in self.abstract_states:
                    probability = 0

                    if abstract_state in ground_mdp.states() and abstract_successor_state in ground_mdp.states():
                        probability = ground_mdp.transition_function(abstract_state, abstract_action, abstract_successor_state)
                    elif abstract_state in ground_mdp.states() and abstract_successor_state in abstract_mdp.states():
                        for ground_successor_state in abstract_mdp.get_ground_states(abstract_successor_state):
                            probability += self.weights[ground_successor_state] * ground_mdp.transition_function(abstract_state, abstract_action, ground_successor_state)
                    elif abstract_state in abstract_mdp.states() and abstract_successor_state in ground_mdp.states():
                        for ground_state in abstract_mdp.get_ground_states(abstract_state):
                            probability += self.weights[ground_state] * ground_mdp.transition_function(ground_state, abstract_action, abstract_successor_state)
                    else:
                        probability = abstract_mdp.transition_function(abstract_state, abstract_action, abstract_successor_state)

                    abstract_transition_probabilities[abstract_state][abstract_action][abstract_successor_state] = probability

        return abstract_transition_probabilities

    def __compute_abstract_start_state_probabilities(self, ground_mdp, abstract_mdp):
        abstract_start_state_probabilities = {}

        for abstract_state in self.abstract_states:
            if abstract_state in ground_mdp.states():
                abstract_start_state_probabilities[abstract_state] = ground_mdp.start_state_function(abstract_state)
            else:
                for ground_state in abstract_mdp.get_ground_states(abstract_state):
                    abstract_start_state_probabilities[abstract_state] = self.weights[ground_state] * ground_mdp.start_state_function(ground_state)

        return abstract_start_state_probabilities

    def __init__(self, ground_mdp, abstract_mdp, abstract_state):
        # TODO: Move this to another location
        self.weights = self.__compute_weights(abstract_mdp)

        self.abstract_states = self.__compute_abstract_states(abstract_mdp, abstract_state)
        self.abstract_actions = ground_mdp.actions()
        self.abstract_rewards = self.__compute_abstract_rewards(ground_mdp, abstract_mdp)
        self.abstract_transition_probabilities = self.__compute_abstract_transition_probabilities(ground_mdp, abstract_mdp)
        self.abstract_start_state_probabilities = self.__compute_abstract_start_state_probabilities(ground_mdp, abstract_mdp)

    def states(self):
        return list(self.abstract_states)

    def actions(self):
        return self.abstract_actions

    def reward_function(self, state, action):
        return self.abstract_rewards[state][action]

    def transition_function(self, state, action, successor_state):
        return self.abstract_transition_probabilities[state][action][successor_state]

    def start_state_function(self, state):
        return self.abstract_start_state_probabilities[state]
