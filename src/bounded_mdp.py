import math


class BoundedMDP:

def __compute_mean_abstract_reward():

def __compute_median_abstract_reward():

def __compute_midpoint_abstract_reward():

# NOTE: This is the simplest thing I could think of. The BMDP paper cites a 1992 paper (Lee and Yannakakis) with aslightly more complicated algorithm.
def __create_new_partition(index, abstract_states):
    concrete_block = abstract_states[index]
    sz = len(concrete_block)
    abstract_states[index] = concrete_block[0:math.floor(sz/2)]
    abstract_states[len(abstract_states)] = concrete_block[math.floor(sz/2):sz]
    return abstract_states

def __check_block_reward_uniformity(mdp, block_states):
    for action in mdp.actions():
        # TODO: Can make more efficient by indexing into block states since abs(a-b) == abs(b-a)
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
                # TODO: Can make more efficient by indexing into block states since abs(a-b) == abs(b-a)
                for p in abstract_states[block_index]:
                    for q in abstract_states[block_index]:
                        if abs(transition_probs[p] - transition_probs[q]) > self.epsilon:
                            return False
    return True
                    
    # NOTE: Not sure if more efficient to check rewards first for all blocks and then transitions for all blocks, or to check both for a single block at a time. Currently doing the latter.
def __check_stability(mdp, abstract_states):
    for block_index in range(len(abstract_states)):
        is_eps_uniform = check_block_reward_uniformity(mdp, abstract_states[block_index])
        if not is_eps_uniform:
            return False, block_index

        is_eps_stable = check_block_transition_stability(mdp, abstract_states, block_index)
        if not is_eps_stable:
            return False, block_index
    
    return True, -1

def __compute_abstract_states(mdp):
    epsilon = min(epsilon, 1.0)
    epsilon = max(epsilon, 0.0)
    abstract_states = {}
    #TODO: Will probably need a different representation if we have an MDP with trillions of states
    concrete_states = mdp.states
    abstract_states[0] = concrete_states[0:len(concrete_states)] # one big block
    abstract_states = create_new_partition(mdp, index, abstract_states) # make a new partition
    eps_stable = False
    while not eps_stable:
        #TODO: Need to add a check that each block can fit in memory
        stable, index = check_stability(mdp, abstract_states)
        if stable:
            eps_stable = True
        else:
            abstract_states = create_new_partition(index, abstract_states)

    if self.bound_type == 'MEAN':
        raise NotImplementedError
    elif self.bound_type == 'MEDIAN':
        raise NotImplementedError
    elif self.bound_type == 'MIDPOINT':
        raise NotImplementedError
    else:
        print("No valid bound_type given (MEAN, MEDIAN, MIDPOINT), defaulting to MEDIAN")
        raise NotImplementedError

    return abstract_states

    def __init__(self, mdp, epsilon, bound_type):
        self.epsilon = epsilon
        self.bound_type = bound_type # TODO this is a bad name
        self.actions = mdp.actions
        self.states = compute_abstract_states(mdp)

    def states(self):
        return self.states

    def actions(self):
        return self.actions

    def transition_function(self, state, action, successor_state):
        raise NotImplementedError

    def reward_function(self, state, action):
        #return self.reward_map[state][action]
        raise NotImplementedError

    def start_state_function(self, state):
        raise NotImplementedError
