import gym
from gym import spaces

# Notes
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


class MetareasoningEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, ):
    super(MetareasoningEnv, self).__init__()
    
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    pass

  def reset(self):
    pass

  def render(self, mode='human', close=False):
    pass
