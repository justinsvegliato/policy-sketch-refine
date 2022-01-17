import gym
from tqdm import tqdm

from metareasoning_dqn_agent import MetareasoningDqnAgent

MODEL_PATH = "models/test-model.pth"

SEED = 1423
EPISODES = 100
ENVIRONMENT = gym.make('reinforcement_learning:metareasoning-v0')
ENVIRONMENT = gym.wrappers.Monitor(ENVIRONMENT, "records", force='True')

INPUT_DIMENSION = ENVIRONMENT.observation_space.shape[0]
HIDDEN_DIMENSION = 64
OUTPUT_DIMENSION = ENVIRONMENT.action_space.n
LEARNING_RATE = 1e-3
SYNC_FREQUENCY = 5
EXPERIENCE_BUFFER_SIZE = 256
AGENT = MetareasoningDqnAgent(seed=SEED, layer_sizes=[INPUT_DIMENSION, HIDDEN_DIMENSION, OUTPUT_DIMENSION], learning_rate=LEARNING_RATE, sync_frequency=SYNC_FREQUENCY, experience_buffer_size=EXPERIENCE_BUFFER_SIZE)


def main():
    AGENT.load_model(MODEL_PATH)

    cumulative_reward_list = []

    for _ in tqdm(range(EPISODES)):
        observation, done, cumulative_reward = ENVIRONMENT.reset(), False, 0

        while not done:
            action = AGENT.get_action(observation, ENVIRONMENT.action_space.n, epsilon=0)
            observation, reward, done, _ = ENVIRONMENT.step(action.item())
            cumulative_reward += reward

        cumulative_reward_list.append(cumulative_reward)

    print("Average Reward Per Episode:", sum(cumulative_reward_list) / len(cumulative_reward_list))


if __name__ == '__main__':
    main()
