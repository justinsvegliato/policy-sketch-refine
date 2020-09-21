class PartiallyAbstractMDP:
    def __compute_weights(self, abstract_mdp):
        weights = {}

        for abstract_state in abstract_mdp.states():
            ground_states = abstract_mdp.get_ground_states([abstract_state])
            for ground_state in ground_states:
                weights[ground_state] = 1 / len(ground_states)

        return weights

    def __compute_states(self, abstract_mdp, grounded_abstract_states):
        ground_states = abstract_mdp.get_ground_states(grounded_abstract_states)
        abstract_states = [abstract_state for abstract_state in abstract_mdp.states() if abstract_state not in grounded_abstract_states]
        all_states = ground_states + abstract_states
        return all_states

    def __compute_abstract_rewards(self, ground_mdp, abstract_mdp):
        rewards = {}

        for state in self.states:
            rewards[state] = {}

            # TODO: Make the lookup in this for loop more efficient
            for action in self.actions:
                # For a ground state, copy the reward from the ground MDP
                if state in ground_mdp.states():
                    rewards[state][action] = ground_mdp.reward_function(state, action)
                # For an abstract state, use the weighted sum of all of its ground states rewards
                else:
                    rewards[state][action] = 0
                    for ground_state in abstract_mdp.get_ground_states([state]):
                        # TODO: Determine whether weights or a max operator should be used here
                        # abstract_rewards[abstract_state][abstract_action] += self.weights[ground_state] * ground_mdp.reward_function(ground_state, abstract_action)
                        rewards[state][action] = max(ground_mdp.reward_function(ground_state, action), rewards[state][action])

        return abstract_rewards

    def __compute_abstract_transition_probabilities(self, ground_mdp, abstract_mdp):
        # TODO: cache all results so that we only ever do this for unique combinations of abstract / ground states

        transition_probabilities = {}

        for state in self.states:
            transition_probabilities[state] = {}

            for action in self.actions:
                transition_probabilities[state][action] = {}

                for successor_state in self.states:
                    probability = 0

                    # Both s and s' are ground states
                    if state in ground_mdp.states() and successor_state in ground_mdp.states():
                        probability = ground_mdp.transition_function(state, action, successor_state)

                    # s is a ground state, s' is an abstract state
                    elif state in ground_mdp.states() and successor_state in abstract_mdp.states():
                        # If transition probability in abstract mdp is zero, then it is also zero for any underlying ground states!
                        if abstract_mdp.transition_function(abstract_mdp.get_abstract_state(state), action, successor_state) > 0:       # comment this and the identical line below for benchmarking
                            for ground_successor_state in abstract_mdp.get_ground_states([successor_state]):
                                probability += ground_mdp.transition_function(state, action, ground_successor_state)
                    
                    # s is an abstract state and s' is a ground state
                    elif state in abstract_mdp.states() and successor_state in ground_mdp.states():
                        # If transition probability in abstract mdp is zero, then it is also zero for any underlying ground states!
                        if abstract_mdp.transition_function(state, action, abstract_mdp.get_abstract_state(successor_state)) > 0:
                            for ground_state in abstract_mdp.get_ground_states([state]):
                                probability += self.weights[ground_state] * ground_mdp.transition_function(ground_state, action, successor_state)
                    
                    # Both s and s' are abstract states
                    else:
                        probability = abstract_mdp.transition_function(state, action, successor_state)

                    abstract_transition_probabilities[state][action][successor_state] = probability

        return abstract_transition_probabilities

    def __compute_start_state_probabilities(self, ground_mdp, abstract_mdp):
        start_state_probabilities = {}

        for state in self.states:
            start_state_probabilities[state] = 0

            if state in ground_mdp.states():
                start_state_probabilities[state] = ground_mdp.start_state_function(state)
            else:
                for ground_state in abstract_mdp.get_ground_states([state]):
                    start_state_probabilities[state] += ground_mdp.start_state_function(ground_state)

        return start_state_probabilities

    def __init__(self, ground_mdp, abstract_mdp, grounding_abstract_states):
        self.weights = self.__compute_weights(abstract_mdp)

        self.states = self.__compute_states(abstract_mdp, grounding_abstract_states)
        self.actions = ground_mdp.actions()
        self.rewards = self.__compute_rewards(ground_mdp, abstract_mdp)
        self.transition_probabilities = self.__compute_transition_probabilities(ground_mdp, abstract_mdp)
        self.start_state_probabilities = self.__compute_start_state_probabilities(ground_mdp, abstract_mdp)

    def states(self):
        return list(self.states)

    def actions(self):
        return self.actions

    def reward_function(self, state, action):
        return self.rewards[state][action]

    def transition_function(self, state, action, successor_state):
        return self.transition_probabilities[state][action][successor_state]

    def start_state_function(self, state):
        return self.start_state_probabilities[state]
