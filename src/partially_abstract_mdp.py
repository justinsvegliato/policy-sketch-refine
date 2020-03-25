class PartiallyAbstractMDP:
    def __compute_abstract_states(self, ground_mdp, abstract_mdp, abstract_state):
        ground_states = abstract_mdp.get_ground_states(abstract_state)
        abstract_states = [state for state in abstract_mdp.states() if state != abstract_state]
        return ground_states + abstract_states


    def __init__(self, ground_mdp, abstract_mdp, abstract_state):
        self.abstract_states = self.__compute_abstract_states(ground_mdp, abstract_mdp, abstract_state)
        self.abstract_actions = ground_mdp.actions()

    def states(self):
        return list(self.abstract_states)

    def actions(self):
        return self.abstract_actions
