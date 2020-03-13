

#TODO: unclear how to compute abstract state "centers". For now, just picking IDs deterministically till I can't grow clusters any more. This is in general not optimal.

#TODO: currently assuming abstract MDP fits within memory.

def compute_abstract_states(mdp, epsilon, bound_type):
    epsilon = min(epsilon, 1.0)
    epsilon = max(epsilon, 0.0)
    abstract_states = {}
    concrete_states = mdp.states
    states_abstracted = 0
    min_state_not_abstracted = 0
    while states_abstracted < len(concrete_states):
        if 
        

    if bound_type == 'MEAN':
        raise NotImplementedError
    elif bound_type == 'MEDIAN':
        raise NotImplementedError
    elif bound_type == 'MIDPOINT':
        raise NotImplementedError
    else:
        print("No valid bound_type given (MEAN, MEDIAN, MIDPOINT), defaulting to MEDIAN")
        raise NotImplementedError

    return abstract_states

class BoundedMDP:
    def __init__(self, mdp, epsilon, bound_type):
        self.states = compute_abstract_states(mdp, epsilon, bound_type)
        self.actions = mdp.actions

    def states(self):
        return self.states

    def actions(self):
        return self.actions

    def transition_function(self, state, action, successor_state):
        raise NotImplementedError

    def reward_function(self, state, action):
        raise NotImplementedError

    def start_state_function(self, state):
        raise NotImplementedError
