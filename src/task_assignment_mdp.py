import itertools as it
from enum import Enum


def power_set(iterable):
    s = list(iterable)
    powerSet = it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))
    powerSetList = [set(ele) for ele in list(powerSet)]
    return powerSetList


class RobotType(Enum):
    TURTLEBOT = 0
    JACKAL = 1
    HUMAN = 2


class Robot(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.loc = None

    def get_name(self):
        return self.name

    def set_loc(self, location):
        self.loc = location
        
    def get_loc(self):
        return self.loc

    def get_cost(self, distance_map, trajectory):
        cost = 0

        for i in range(len(trajectory) - 1):
            cost += distance_map[trajectory[i]][trajectory[i + 1]]

        if self.type == RobotType.TURTLEBOT:
            return 2 * cost

        return cost


class Task(object):
    def __init__(self, id, start_time, end_time):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time


class RTAMDP(object):
    def _compute_states(self):
        times = [i for i in range(self.horizon + 1)]
        robots = [list(robot) for robot in list(it.product([0, 1], repeat=len(self.robots) - 1))]
        for robot in robots:
            # The final agent (human) can never break.
            robot.append(1)
        return list(it.product(times, list(power_set(self.tasks)), robots))

    def _compute_actions(self):
        actions = list(power_set(list(it.product(self.tasks, self.robots))))
        actions = [action for action in actions if len(action) <= len(self.tasks)]

        new_actions = []
        for action in actions:
            task_check = [0 for _ in self.tasks]
            robot_check = [0 for _ in self.robots]

            check = True

            for (task, robot) in action:
                if task_check[self.tasks.index(task)] == 1:
                    check = False
                    break
                elif robot_check[self.robots.index(robot)] == 1:
                    check = False
                    break
                else:
                    task_check[self.tasks.index(task)] = 1
                    robot_check[self.robots.index(robot)] = 1

            if check is True:
                new_actions.append(action)

        return new_actions

    # def _compute_state_transitions(self):
    #     S = [[[int(-1) for sp in range(self.mdp.ns)]
    #           for a in range(self.mdp.m)] for s in range(self.mdp.n)]
    #     T = [[[float(0.0) for sp in range(self.mdp.ns)]
    #           for a in range(self.mdp.m)] for s in range(self.mdp.n)]

    #     for s, state in enumerate(self.states):

    #         for a, action in enumerate(self.actions):

    #             if state[0] == self.horizon:
    #                 T[s][a][s] == 1.0
    #                 break

    #             tasks = []
    #             robots = []
    #             invalid = False
    #             for (task, robot) in action:
    #                 if task[0] > state[0]:  # Task cannot yet be completed
    #                     invalid = True
    #                     break
    #                 tasks.append(task)
    #                 robots.append(robot)

    #             for sp, statePrime in enumerate(self.states):
    #                 S[s][a][sp] = sp

    #                 if statePrime[0] == state[0] + 1:

    #                     # If the action set is empty or there are no tasks or the action is invalid
    #                     # then only one transition is possible.
    #                     if len(action) == 0 or len(state[1]) == 0 or invalid == True:
    #                         if statePrime[1] == state[1] and statePrime[2] == state[2]:
    #                             T[s][a][sp] = 1.0
    #                             break
    #                         else:
    #                             continue

    #                     if state[1] == statePrime[1].union(tasks):
    #                         for (task, robot) in action:
    #                             # Make sure a broken robot is not assigned a task
    #                             if state[2][robot.get_id()] == 0:
    #                                 T[s][a][sp] = 0.0
    #                                 break
    #                             # Make sure not assigned completed tasks
    #                             if task not in state[1]:
    #                                 T[s][a][sp] = 0.0
    #                                 break
    #                             # Only assign possible tasks
    #                             if task.start_time > state[0] or task.end_time < statePrime[0]:
    #                                 T[s][a][sp] = 0.0
    #                                 break
    #                             # If robot breaks
    #                             if state[2][robot.get_id()] == 1 and statePrime[2][robot.get_id()] == 0:
    #                                 # And the task was not erroneously completed
    #                                 if task in state[1] and task in statePrime[1]:
    #                                     # Update the transition probability
    #                                     if T[s][a][sp] == 0:
    #                                         T[s][a][sp] = robot.get_break_probability(
    #                                             task.start_location, task.end_location)
    #                                     else:
    #                                         T[s][a][sp] = T[s][a][sp]*robot.get_break_probability(
    #                                             task.start_location, task.end_location)
    #                                 else:
    #                                     T[s][a][sp] = 0.0
    #                                     break
    #                             # If robot doesn't break
    #                             if state[2][robot.get_id()] == 1 and statePrime[2][robot.get_id()] == 1:
    #                                 # And task was completed
    #                                 if task in state[1] and task not in statePrime[1]:
    #                                     # Update the transition probability
    #                                     if T[s][a][sp] == 0.0:
    #                                         T[s][a][sp] = (
    #                                             1-robot.get_break_probability(task.start_location, task.end_location))
    #                                     else:
    #                                         T[s][a][sp] = T[s][a][sp]*(1-robot.get_break_probability(
    #                                             task.start_location, task.end_location))
    #                                 else:
    #                                     T[s][a][sp] = 0.0
    #                                     break

    #                         # Ensure that the robot status list is consistent
    #                         for i in range(len(state[2])):
    #                             # Make sure no broken robot is suddenly set to working
    #                             if state[2][i] == 0 and statePrime[2][i] == 1:
    #                                 T[s][a][sp] = 0.0
    #                             # Make sure that if a robot breaks it is in action
    #                             if state[2][i] == 1 and statePrime[2][i] == 0:
    #                                 r_check = False
    #                                 for (task, robot) in action:
    #                                     if robot.get_id() == i:
    #                                         r_check = True
    #                                 if not r_check:
    #                                     T[s][a][sp] = 0.0

    #             # When T[s][a] is zero everywhere make it go w.p. 1 to same state with time = time + 1
    #             if np.sum(T[s][a]) == 0.0:
    #                 for sp, statePrime in enumerate(self.states):
    #                     if statePrime[0] == (state[0]+1) and statePrime[1] == state[1] and statePrime[2] == state[2]:
    #                         T[s][a][sp] = 1.0
    #                         break

    #     return S, T

    # def _compute_rewards(self, T):
    #     R = [[0.0 for a in range(self.mdp.m)] for s in range(self.mdp.n)]
    #     R_full = [[[0.0 for sp in range(self.mdp.ns)] for a in range(
    #         self.mdp.m)] for s in range(self.mdp.n)]

    #     for s, state in enumerate(self.states):
    #         for a, action in enumerate(self.actions):
    #             for sp, statePrime in enumerate(self.states):
    #                 reward = 0
    #                 for task in statePrime[1]:
    #                     # If available task is not completed, punish for length of action time (delta)
    #                     if task.start_time < statePrime[0]:
    #                         reward -= self.duration
    #                 for (task, robot) in action:
    #                     # Action is invalid so skip because T is 0
    #                     if task not in state[1]:
    #                         continue
    #                     # Robot breaks
    #                     elif task in statePrime[1]:
    #                         reward -= 100
    #                     # Robot successfully completes the task
    #                     else:
    #                         reward -= robot.calculate_time(
    #                             task.start_location, task.end_location)
    #                 R[s][a] += T[s][a][sp] * reward
    #                 R_full[s][a][sp] = reward
    #     return (R, R_full)

    # def _compute_start_state_probabilities(self):
    #     return None

    def __init__(self, tasks, robots, horizon=4, duration=30):
        self.tasks = tasks
        self.robots = robots
        self.horizon = horizon
        self.duration = duration

        self.state_space = self._compute_states()
        self.action_space = self._compute_actions()
        # self.transition_probabilities = self.compute_transition_probabilities()
        # self.reward = self._compute_rewards()
        # self.start_state_probabilities = self._compute_start_state_probabilities()

    def states(self):
        return list(self.state_space)

    def actions(self):
        return self.action_space

    # def reward_function(self, state, action):
    #     return self.rewards[state][action]

    # def transition_function(self, state, action, successor_state):
    #     return self.transition_probabilities[state][action][successor_state]

    # def start_state_function(self, state):
    #     return self.start_state_probabilities[state]

def main():
    tasks = [Task(0, 2, 1), Task(1, 1, 0), Task(2, 1, 2), Task(3, 2, 3)]
    rewards = [Robot("Matteo", 0), Robot("Samer", 1), Robot("Connor", 2)]
    mdp = RTAMDP(tasks, rewards)

    print(mdp.states())

main()
