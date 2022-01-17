
import logging
import time

import gym
import numpy as np
from gym import spaces

import policy_sketch_refine
import utils
from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = (6, 3)
POINTS_OF_INTEREST = 2
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

HORIZON = 100
SIMULATIONS = 100

EXPANSION_STRATEGY_MAP = {
  0: 'NAIVE',
  1: 'GREEDY',
  2: 'PROACTIVE'
}

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


class MetareasoningEnv(gym.Env):

  def __init__(self, ):
    super(MetareasoningEnv, self).__init__()
    
    self.observation_space = spaces.Box(low=np.array([np.float32(0.0), ]), high=np.array([np.float32(1.0), ]))
    self.action_space = spaces.Discrete(3)

    self.ground_mdp = None
    self.abstract_mdp = None

    self.current_ground_state = None
    self.current_action = None
    self.current_abstract_state = None

    self.state_history = []
    self.ground_policy = {}
    self.solved_ground_states = []

    self.steps = 0

  def step(self, action):    
    logging.info("Encountered a new abstract state: [%s]", self.current_abstract_state)

    logging.info("Starting the policy sketch refine algorithm...")
    start = time.time()
    solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPAND_POINTS_OF_INTEREST, EXPANSION_STRATEGY_MAP[action], GAMMA)
    logging.info("Finished the policy sketch refine algorithm: [time=%f]", time.time() - start)

    start = time.time()
    values = utils.get_ground_entities(solution['values'], self.ground_mdp, self.abstract_mdp)
    logging.info("Calculated the values from the solution of policy sketch refine: [time=%f]", time.time() - start)

    start = time.time()
    ground_states = self.abstract_mdp.get_ground_states([self.current_abstract_state])
    logging.info("Calculated the ground states for the current abstract state: [time=%f]", time.time() - start)

    start = time.time()
    policy = utils.get_ground_policy(values, self.ground_mdp, self.abstract_mdp, ground_states, self.current_abstract_state, GAMMA)
    logging.info("Calculated the policy from the values: [time=%f]", time.time() - start)

    logging.info("Cached the ground states for the new abstract state: [%s]", self.current_abstract_state)
    self.solved_ground_states += ground_states
    for ground_state in ground_states:
      self.ground_policy[ground_state] = policy[ground_state]

    while self.current_ground_state in self.solved_ground_states:
      self.state_history.append(self.current_ground_state)

      self.current_action = self.ground_policy[self.current_ground_state]

      logging.info("Current Ground State: [%s]", self.current_ground_state)
      logging.info("Current Abstract State: [%s]", self.current_abstract_state)
      logging.info("Current Action: [%s]", self.current_action)

      self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
      self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    
      self.steps += 1

    return self.__get_observation(), self.__get_reward(), self.__get_done(), None

  def reset(self):
    start = time.time()
    self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()), time.time() - start)

    start = time.time()
    self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()), time.time() - start)  

    start = time.time()
    abstract_solution = policy_sketch_refine.sketch(self.abstract_mdp, GAMMA)
    abstract_policy = utils.get_full_ground_policy(abstract_solution['values'], self.abstract_mdp, self.abstract_mdp.states(), GAMMA)
    logging.info("Solved the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()), time.time() - start)

    self.current_ground_state = INITIAL_GROUND_STATE
    self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    logging.info("Initialized the current ground state: [%s]", self.current_ground_state)
    logging.info("Initialized the current abstract state: [%s]", self.current_abstract_state)

    self.state_history = []
    self.solved_ground_states = []

    self.ground_policy = {}
    for ground_state in self.ground_mdp.states():
      self.ground_policy[ground_state] = abstract_policy[self.abstract_mdp.get_abstract_state(ground_state)]

    logging.info("Activating the simulator...")

    return self.__get_observation()

  # TODO Implement policy evaluation because it will still be efficient
  def __get_simulated_cumulative_reward(self):
    simulated_cumulative_rewards = []

    for _ in range(SIMULATIONS):
      simulated_cumulative_reward = 0

      simulated_ground_state = INITIAL_GROUND_STATE
      for _ in range(HORIZON):
          simulated_action = self.ground_policy[simulated_ground_state]

          simulated_reward = self.ground_mdp.reward_function(simulated_ground_state, simulated_action)
          simulated_cumulative_reward += simulated_reward

          simulated_ground_state = utils.get_successor_state(simulated_ground_state, simulated_action, self.ground_mdp)
        
      simulated_cumulative_rewards.append(simulated_cumulative_reward)

    return sum(simulated_cumulative_rewards) / len(simulated_cumulative_rewards)

  # TODO Improve this function because it can be more accurate/efficient
  def __get_maximum_cumulative_reward(self):
    states = self.ground_mdp.states()
    actions = self.ground_mdp.actions()

    maximum_immediate_reward = float('-inf')
    for state in states:
        for action in actions:
            move_reward = self.ground_mdp.reward_function(state, action)
            maximum_immediate_reward = max(maximum_immediate_reward, move_reward)

    return HORIZON * maximum_immediate_reward

  def __get_observation(self):
    quality = self.__get_simulated_cumulative_reward() / self.__get_maximum_cumulative_reward() 
    return np.array([np.float32(quality),])

  def __get_reward(self):
    return self.__get_maximum_cumulative_reward() - self.__get_simulated_cumulative_reward()

  def __get_done(self):
    return self.steps >= HORIZON
  

def main():
  env = MetareasoningEnv()

  print(env.reset())
  print(env.step(2))


if __name__ == '__main__':
  main()
