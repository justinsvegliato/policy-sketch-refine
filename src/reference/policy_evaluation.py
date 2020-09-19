import math

# Returns a set of values for each state, given the policy.
def evaluatePolicy(mdp, policy, gamma, eps):
  values = {}
  for state in mdp.states():
    values[state] = 0.0
  
  delta = 0.0
  while True:
    for state in mdp.states():
      action = policy[state]
      immediate_reward = mdp.reward_function(state, action)
      expected_future_reward = 0
      for successor_state in mdp.states():
        expected_future_reward += mdp.transition_function(state, action, successor_state) * values[successor_state]
      new_value = immediate_reward + gamma * expected_future_reward
      delta = max(delta, math.fabs(new_value - value[state]))

    if delta < eps:
      return values

