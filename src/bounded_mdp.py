import math
import statistics

class BoundedMDP:
    def __compute_mean_abstract_reward(self, mdp):
        reward = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                block_reward_given_action = 0.0
                for ground_state in block_states:
                    block_reward_given_action += mdp.reward_function(ground_state, action)
                reward[block_index][action] = block_reward_given_action / float(len(block_states))

        return reward

    def __compute_median_abstract_reward(self, mdp):
        reward = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                all_rewards_given_action = []
                for ground_state in block_states:
                    all_rewards_given_action.append(mdp.reward_function(ground_state, action))
                reward[block_index][action] = statistics.median(all_rewards_given_action)

        return reward

    def __compute_midpoint_abstract_reward(self, mdp):
        reward = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                min_reward_given_action = math.inf
                max_reward_given_action = -math.inf
                for ground_state in block_states:
                    reward_given_action = mdp.reward_function(ground_state, action)
                    min_reward_given_action = min(min_reward_given_action, reward_given_action)
                    max_reward_given_action = max(max_reward_given_action, reward_given_action)
                reward[block_index][action] = (min_reward_given_action + max_reward_given_action) / 2.0

        return reward

    def __compute_mean_abstract_transition(self, mdp):
        transition = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                for other_block in range(len(self.states)):
                    other_block_states = self.states[other_block]
                    transition_prob = 0.0
                    for p in block_states:
                        for q in other_block_states:
                            transition_prob += mdp.transition_function(p, action, q)
                    transition[block_index][action][other_block] = transition_prob / float(len(block_states))

        return transition

    def __compute_median_abstract_transition(self, mdp):
        transition = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                for other_block in range(len(self.states)):
                    other_block_states = self.states[other_block]
                    transition_probs = []
                    for p in block_states:
                        for q in other_block_states:
                            transition_probs.append(mdp.transition_function(p, action, q))
                    transition[block_index][action][other_block] = statistics.median(transition_probs)

        return transition

    def __compute_midpoint_abstract_transition(self, mdp):
        transition = {}
        for block_index in range(len(self.states)):
            block_states = self.states[block_index]
            for action in mdp.actions: # could also be self.actions
                for other_block in range(len(self.states)):
                    other_block_states = self.states[other_block]
                    min_trans_prob = math.inf
                    max_trans_prob = -math.inf
                    for p in block_states:
                        for q in other_block_states:
                            transition_prob = mdp.transition_function(p, action, q)
                            min_trans_prob = min(min_trans_prob, transition_prob)
                            max_trans_prob = max(max_trans_prob, transition_prob)
                    transition[block_index][action][other_block] = (min_trans_prob + max_trans_prob) / 2.0

        return transition

    # NOTE This is the simplest thing I could think of.
    # The BMDP paper cites a 1992 paper (Lee and Yannakakis) with aslightly more complicated algorithm.
    def __create_new_partition(index, abstract_states):
        concrete_block = abstract_states[index]
        sz = len(concrete_block)
        abstract_states[index] = concrete_block[0:math.floor(sz/2)]
        abstract_states[len(abstract_states)] = concrete_block[math.floor(sz/2):sz]
        return abstract_states

    def __check_block_reward_uniformity(mdp, block_states):
        for action in mdp.actions():
            # TODO Can make more efficient by indexing into block states since abs(a - b) == abs(b - a)
            for p in block_states:
                for q in block_states:
                    if abs(mdp.reward_function(p, action) - mdp.reward_function(q, action)) > self.epsilon:
                        return False
        return True

    def __check_block_transition_stability(mdp, abstract_states, block_index):
        for other_block in range(len(abstract_states)):
            if other_block != block_index:
                for action in mdp.actions():
                    # Sum transition probabilities and check that they are within epsilon
                    transition_probs = {}
                    for p in abstract_states[block_index]:
                        transition_probability = 0.0
                        for r in abstract_states[other_block]:
                            transition_probability += mdp.transition_function(p, action, r)
                        transition_probs[p] = transition_probability
                    # TODO: Can be more efficient by indexing into block states since abs(a-b) == abs(b-a)
                    for p in abstract_states[block_index]:
                        for q in abstract_states[block_index]:
                            if abs(transition_probs[p] - transition_probs[q]) > self.epsilon:
                                return False
        return True

    # NOTE Not sure if more efficient to check rewards first for all blocks and then
    # transitions for all blocks, or to check both for a single block at a time. 
    # Currently doing the latter.
    def __check_stability(mdp, abstract_states):
        for block_index in range(len(abstract_states)):
            is_eps_uniform = self.__check_block_reward_uniformity(mdp, abstract_states[block_index])
            if not is_eps_uniform:
                return False, block_index

            is_eps_stable = self.__check_block_transition_stability(mdp, abstract_states, block_index)
            if not is_eps_stable:
                return False, block_index

        return True, -1

    def __compute_abstract_states(self, mdp):
        self.epsilon = min(self.epsilon, 1.0)
        self.epsilon = max(self.epsilon, 0.0)

        abstract_states = {}

        # TODO Will probably need a different representation if we have an MDP with trillions of states
        concrete_states = mdp.states()
        abstract_states[0] = concrete_states[0:len(concrete_states)] # one big block
        abstract_states = self.__create_new_partition(mdp, index, abstract_states) # make a new partition
        eps_stable = False
        while not eps_stable:
            # TODO Need to add a check that each block can fit in memory
            stable, index = self.__check_stability(mdp, abstract_states)
            if stable:
                eps_stable = True
            else:
                abstract_states = self.__create_new_partition(index, abstract_states)

        return abstract_states

    def __compute_abstract_rewards(self, mdp):
        if self.bound_type == 'MEAN':
            return self.__compute_mean_abstract_reward(mdp)

        if self.bound_type == 'MEDIAN':
            return self.__compute_median_abstract_reward(mdp)

        if self.bound_type == 'MIDPOINT':
            return self.__compute_midpoint_abstract_reward(mdp)

        print("No valid bound_type given (MEAN, MEDIAN, MIDPOINT), defaulting to MEAN")
        return self.__compute_mean_abstract_reward(mdp)

    # NOTE: The new transition probabilities are not in general normalized. We need to normalize them.
    def __compute_abstract_transitions(self, mdp):
        if self.bound_type == 'MEAN':
            return self.__compute_mean_abstract_transition(mdp)

        if self.bound_type == 'MEDIAN':
            return self.__compute_median_abstract_transition(mdp)

        if self.bound_type == 'MIDPOINT':
            return self.__compute_midpoint_abstract_transition(mdp)

        print('No valid bound_type given (MEAN, MEDIAN, MIDPOINT), defaulting to MEAN')
        return self.__compute_mean_abstract_transition(mdp)

    def __init__(self, mdp, epsilon, bound_type):
        self.epsilon = epsilon
        self.bound_type = bound_type # TODO this is a bad name
        self.actions = mdp.actions
        self.states = self.__compute_abstract_states(mdp)
        self.rewards = self.__compute_abstract_rewards(mdp)
        self.transitions = self.__compute_abstract_transitions(mdp)

    def states(self):
        return self.states

    def actions(self):
        return self.actions

    def transition_function(self, state, action, successor_state):
        return self.transitions[state][action][successor_state] # state IDs for abstract states

    def reward_function(self, state, action):
        return self.rewards[state][action] # state ID for abstract states

    def start_state_function(self, state):
        raise NotImplementedError


def main():
    from grid_world_mdp import GridWorldMDP

    grid_world = [
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'W', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'W', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'W', 'O', 'O', 'O', 'W', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'O', 'O'],
        ['O', 'O', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'W', 'W', 'W', 'W', 'W', 'G', 'O']
    ]
    grid_world_mdp = GridWorldMDP(grid_world)

    bounded_mdp = BoundedMDP(grid_world_mdp, 0.1, 'MEAN')


if __name__ == '__main__':
    main()
