import gym
from gym import spaces


class MetareasoningEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2):
    super(MetareasoningEnv, self).__init__()
    
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    pass

  def reset(self):
    pass

  def render(self, mode='human', close=False):
    pass
