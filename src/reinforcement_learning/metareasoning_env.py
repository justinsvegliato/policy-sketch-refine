
import logging
import time

import gym
from gym import spaces

import policy_sketch_refine
import utils

from earth_observation_abstract_mdp import EarthObservationAbstractMDP
from earth_observation_mdp import EarthObservationMDP

SIZE = (12, 24)
POINTS_OF_INTEREST = 3
VISIBILITY = None

ABSTRACTION = 'MEAN'
ABSTRACT_STATE_WIDTH = 3
ABSTRACT_STATE_HEIGHT = 3

INITIAL_GROUND_STATE = 0

EXPAND_POINTS_OF_INTEREST = True
GAMMA = 0.99

HORIZON = 100

SIMULATIONS = 10

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


class MetareasoningEnv(gym.Env):
  # TODO: Confirm what this does
  metadata = {'render.modes': ['human']}

  def __init__(self, ):
    super(MetareasoningEnv, self).__init__()
    
    # TODO: Adjust the observation space and the action space
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.uint8)
    self.action_space = spaces.Discrete(3)

    self.ground_mdp = None
    self.abstract_mdp = None

    self.current_ground_state = None
    self.current_action = None
    self.current_abstract_state = None

    self.state_history = []
    self.policy_cache = {}

    self.steps = 0

  def step(self, action):    
    logging.info("Encountered a new abstract state: [%s]", self.current_abstract_state)

    # TODO: Modify this code segment to reflect the action parameter of this function 
    logging.info("Starting the policy sketch refine algorithm...")
    start = time.time()
    solution = policy_sketch_refine.solve(self.ground_mdp, self.current_ground_state, self.abstract_mdp, self.current_abstract_state, EXPAND_POINTS_OF_INTEREST, GAMMA)
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
    for ground_state in ground_states:
      self.policy_cache[ground_state] = policy[ground_state]

    while self.current_ground_state in self.policy_cache:
      self.state_history.append(self.current_ground_state)

      self.current_action = self.policy_cache[self.current_ground_state]

      logging.info("Current Ground State: [%s]", self.current_ground_state)
      logging.info("Current Abstract State: [%s]", self.current_abstract_state)
      logging.info("Current Action: [%s]", self.current_action)

      self.current_ground_state = utils.get_successor_state(self.current_ground_state, self.current_action, self.ground_mdp)
      self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    
      self.steps += 1

    return self.__get_observation(), self.__get_reward(), self.__get_done(), self.__get_info()

  def reset(self):
    start = time.time()
    self.ground_mdp = EarthObservationMDP(SIZE, POINTS_OF_INTEREST, VISIBILITY)
    logging.info("Built the earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.ground_mdp.states()), len(self.ground_mdp.actions()), time.time() - start)

    start = time.time()
    self.abstract_mdp = EarthObservationAbstractMDP(self.ground_mdp, ABSTRACTION, ABSTRACT_STATE_WIDTH, ABSTRACT_STATE_HEIGHT)
    logging.info("Built the abstract earth observation MDP: [states=%d, actions=%d, time=%f]", len(self.abstract_mdp.states()), len(self.abstract_mdp.actions()), time.time() - start)

    self.current_ground_state = INITIAL_GROUND_STATE
    self.current_abstract_state = self.abstract_mdp.get_abstract_state(self.current_ground_state)
    logging.info("Initialized the current ground state: [%s]", self.current_ground_state)
    logging.info("Initialized the current abstract state: [%s]", self.current_abstract_state)

    self.state_history = []
    self.policy_cache = {}

    logging.info("Activating the simulator...")

  def render(self, mode='human', close=False):
    pass

  # TODO: Consider doing policy evaluation since it should still be fast
  def __get_simulated_cumulative_reward(self):
    simulated_cumulative_rewards = []

    for _ in range(SIMULATIONS):
      simulated_cumulative_reward = 0

      simulated_ground_state = INITIAL_GROUND_STATE
      for _ in range(HORIZON):
          simulated_action = self.policy_cache[simulated_ground_state]

          simulated_reward = self.ground_mdp.reward_function(simulated_ground_state, simulated_action)
          simulated_cumulative_reward += simulated_reward

          simulated_ground_state = utils.get_successor_state(simulated_ground_state, simulated_action, self.ground_mdp)
        
      simulated_cumulative_rewards.append(simulated_cumulative_reward)

    return simulated_cumulative_rewards / len(simulated_cumulative_rewards)

  # TODO: Improve this function: it can be more accurate and more efficient
  def __get_maximum_cumulative_reward(self):
    states = self.ground_mdp.states()
    actions = self.ground_mdp.actions()

    maximum_immediate_reward = float('-inf')
    for state in states:
        for action in actions:
            move_reward = self.ground_mdp.reward_function(state, action)
            maximum_immediate_reward = max(maximum_immediate_reward, move_reward)

    return HORIZON * maximum_immediate_reward

  # TODO: Add additional features
  def __get_observation(self):
    quality = self.__get_simulated_cumulative_reward / self.__get_maximum_cumulative_reward() 
    return (quality,)

  def __get_reward(self):
    return self.__get_maximum_cumulative_reward() - self.__get_simulated_cumulative_reward()

  def __get_done(self):
    return self.steps >= HORIZON

  def __get_info(self):
    return None


# NOTES
# (1) Let's assume that a single episode is a fixed number of time steps.
# (2) At each point in which we encounter an unvisited abstract state, we need to decide how to make a PAMDP to get the actions for that unvisited abstract state.
# (3) What are the ways in which we can decide to make a PAMDP?
# (4) At a fine-grained level, we must decide which abstract states to expand.
# (5) At a coarse-grained level, we must decide which expansion strategy to use.

# (1) At the start of an episode, what has to happen?
# (2) We have to initalize a random ground MDP and, from that, an abstract MDP among a few other bookkeeping things.
# (3) Does this encapsulate the reset() function? I'd say so. This is what the reset() function ought to do.

# (1) We know the actions of the PAMDP but what should the states be?
# (2) We'll at least have solution quality and time of the policy.
# (3) I'm not exactly sure what time means in this case.
# (4) Solution quality refers to the quality of the policy.
# (5) Is time how long we have solved for already? Maybe we don't need it because we're not using anytime algorithms. It may be helpful though.
# (6) What are other state factors that can be used?
# (7) I guess we need two things:
# (8) First, we need factors that describe the current status of expansion. What have we done already? This may be useful.
# (9) This would likely be a list of abstract states that have been expanded already.
# (10) Second, we need factors that describe why you would expand one state over another state!
# (11) Maybe our state factors could have information regarding each abstract state.
# (12) For each abstract state, we have whether or not we expanded that abstract state, whether or not that abstract state contains reward, 
# whether or not that abstract state touches an abstract state with reward, the distance from the current location to that abstract state, and 
# the number of abstract states that have been expanded already. 
# (13) This seems very reasonable to me. It could probably learn what to do.

# (14) For an initial round of implementation, let's implement solution quality, and the number of abstract states that have been expanded.
# (15) For the actions, let's use expansion strategies? I guess? It's honestly easier to implement no expansion strategys.

# (16) How should the step function work?
# (17) At the minimum, the step function must make a PAMDP. 
# (18) You have two perspectives: 
# (a) You make a series of actions in the world (through visited abstract states that contain ground states) until you encounter an unvisited abstract state
# (b) You encounter an unvisited abstract state and then make a series of actions in the world (through visited abstract states that contain ground states)
# (19) In other words, you either expand an unvisited state and then act or act and then expand an unvisited state
# (20) Since we need a reward, I think expanding an unvisited state and then acting is the way to go because you'll get reward that perhaps reflects that decision (or not)