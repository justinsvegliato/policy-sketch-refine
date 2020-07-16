import itertools as it
import numpy as np

from enum import Enum
from IPython import embed


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
    def __init__(self, name, ID, type):
        self.name = name
        self.id = ID
        self.type = type
        self.loc = None

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

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

    def get_break_probability(self):
        return 0.0 if self.type == 2 else 0.1

    def calculate_time(self, loc1, loc2):
        # This is currently a dummy value
        return 1.0


class Task(object):
    def __init__(self, ID, start_time, end_time, start_location, end_location):
        self.id = ID
        self.start_time = start_time
        self.end_time = end_time
        self.start_location = start_location
        self.end_location = end_location
        assert(self.end_time > self.start_time)
        assert(self.start_location != self.end_location)


class RTAMDP(object):
    def __init__(self, tasks, robots, horizon=4, duration=30):
        self.tasks = tasks
        self.robots = robots
        self.horizon = horizon
        self.duration = duration

        self._state_space = self._compute_states()
        self._action_space = self._compute_actions()
        self._transitions = self._compute_transitions()
        self._rewards = self._compute_rewards()

        self.check_validity()


    @property
    def states(self):
        return self._state_space

    @property
    def actions(self):
        return self._action_space

    @property
    def transitions(self):
        return self._transitions

    @property
    def rewards(self):
        return self._rewards

    def _compute_states(self):
        '''
        A state is a list of [time, [list of available tasks], [status of each robot]]
        '''
        times = [i for i in range(self.horizon + 1)]
        robot_statuses = [list(robot_status) + [1] for robot_status in list(it.product([0, 1], repeat=len(self.robots) - 1))]
        return list(it.product(times, list(power_set(self.tasks)), robot_statuses))

    def _compute_actions(self):
        '''
        An action is an assignement of available tasks to robots.
        '''
        actions = []
        # First create *all* possible assignments, regardless of validity.
        tmp_actions = list(power_set(list(it.product(self.tasks, self.robots))))
        # Prune any action that assigns more tasks than we have.
        tmp_actions = [action for action in tmp_actions if len(action) <= len(self.tasks)]
        # embed()
        while len(tmp_actions) > 0:
            action = tmp_actions.pop()
            # print(action)
            check_set_tasks = set()
            check_set_robots = set()
            for assignment in action:
                check_set_tasks.add(assignment[0])
                check_set_robots.add(assignment[1])
            # print(action)
            if len(check_set_tasks) == len(action) and len(check_set_robots) == len(action):
                actions.append(action)

        return actions

    def _compute_transitions(self):
        T = np.zeros((len(self.states), len(self.actions), len(self.states)))

        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):

                if action == set():
                    T[s][a][s] = 1.0
                    continue

                if state[0] == self.horizon:
                    T[s][a][s] = 1.0
                    continue

                tasks = []
                robots = []
                invalid = False
                for (task, robot) in action:
                    if task.start_time > state[0]:  # Task cannot yet be completed
                        invalid = True
                        break
                    tasks.append(task)
                    robots.append(robot)

                for sp, statePrime in enumerate(self.states):
                    if statePrime[0] == state[0] + 1:
                        # If the action set is empty or there are no tasks or the action is invalid
                        # then only one transition is possible.
                        if len(action) == 0 or len(state[1]) == 0 or invalid == True:
                            if statePrime[1] == state[1] and statePrime[2] == state[2]:
                                T[s][a][sp] = 1.0
                                break
                            else:
                                continue

                        if state[1] == statePrime[1].union(tasks):
                            for (task, robot) in action:
                                # Make sure a broken robot is not assigned a task
                                if state[2][robot.get_id()] == 0:
                                    T[s][a][sp] = 0.0
                                    break
                                # Make sure not assigned completed tasks
                                if task not in state[1]:
                                    T[s][a][sp] = 0.0
                                    break
                                # Only assign possible tasks
                                if task.start_time > state[0] or task.end_time < statePrime[0]:
                                    T[s][a][sp] = 0.0
                                    break
                                # If robot breaks
                                if state[2][robot.get_id()] == 1 and statePrime[2][robot.get_id()] == 0:
                                    # And the task was not erroneously completed
                                    if task in state[1] and task in statePrime[1]:
                                        # Update the transition probability
                                        if T[s][a][sp] == 0:
                                            T[s][a][sp] = robot.get_break_probability()
                                        else:
                                            T[s][a][sp] = T[s][a][sp]*robot.get_break_probability()
                                    else:
                                        T[s][a][sp] = 0.0
                                        break
                                # If robot doesn't break
                                if state[2][robot.get_id()] == 1 and statePrime[2][robot.get_id()] == 1:
                                    # And task was completed
                                    if task in state[1] and task not in statePrime[1]:
                                        # Update the transition probability
                                        if T[s][a][sp] == 0.0:
                                            T[s][a][sp] = (
                                                1-robot.get_break_probability())
                                        else:
                                            T[s][a][sp] = T[s][a][sp]*(1-robot.get_break_probability())
                                    else:
                                        T[s][a][sp] = 0.0
                                        break

                            # Ensure that the robot status list is consistent
                            for i in range(len(state[2])):
                                # Make sure no broken robot is suddenly set to working
                                if state[2][i] == 0 and statePrime[2][i] == 1:
                                    T[s][a][sp] = 0.0
                                # Make sure that if a robot breaks it is in action
                                if state[2][i] == 1 and statePrime[2][i] == 0:
                                    r_check = False
                                    for (task, robot) in action:
                                        if robot.get_id() == i:
                                            r_check = True
                                    if not r_check:
                                        T[s][a][sp] = 0.0

                # When T[s][a] is zero everywhere make it go w.p. 1 to same state with time = time + 1
                if np.sum(T[s][a]) == 0.0:
                    for sp, statePrime in enumerate(self.states):
                        if statePrime[0] == (state[0]+1) and statePrime[1] == state[1] and statePrime[2] == state[2]:
                            T[s][a][sp] = 1.0
                            break
        return T

    def check_validity(self):
        """
            params: 
                None

            returns:
                None

            description:
                Checks to make sure that T is a proper probability distribution.
                If there is some (s,a) for which T[s][a] does not sum to one,
                enter embed for debugging purposes and then quit.
        """
        for s in range(len(self.states)):
            for a in range(len(self.actions)):

                if round(np.sum(self._transitions[s][a]), 3) != 1.0:
                    print("Error @ state " + str(self.states[s]) + " and action " + str(self.actions[a]))
                    embed()
                    quit()
        print("Transition function is valid.")

    def _compute_rewards(self):
        R = np.zeros((len(self.states), len(self.actions)))

        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                for sp, statePrime in enumerate(self._state_space):
                    reward = 0
                    for task in statePrime[1]:
                        # If available task is not completed, punish for length of action time (delta)
                        if task.start_time < statePrime[0]:
                            reward -= self.duration
                    for (task, robot) in action:
                        # Action is invalid so skip because T is 0
                        if task not in state[1]:
                            continue
                        # Robot breaks
                        elif task in statePrime[1]:
                            reward -= 10
                        # Robot successfully completes the task
                        else:
                            reward -= robot.calculate_time(task.start_location, task.end_location)
                    R[s][a] += self.transitions[s][a][sp] * reward
        return R

    def _compute_start_state_probabilities(self):
        return np.ones((len(self.states),)) / len(self.states)

    def reward_function(self, state, action):
        return self._rewards[state][action]

    def transition_function(self, state, action, successor_state):
        return self._transitions[state][action][successor_state]

    def start_state_probability(self, state):
        return self._start_state_probabilities[state]

def main():
    tasks = [Task(0, 1, 2, 'loc1', 'loc2'), Task(1, 0, 1, 'loc3', 'loc4'),
             Task(2, 1, 2, 'loc5', 'loc6'), Task(3, 2, 3, 'loc7', 'loc8')]
    rewards = [Robot("Matteo", 0, 0), Robot("Samer", 1, 1), Robot("Connor", 2, 2)]
    mdp = RTAMDP(tasks, rewards)
    print(len(mdp.states), len(mdp.actions))
if __name__ == '__main__':
    main()