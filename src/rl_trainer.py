from tqdm import tqdm

from rl_agent import RLAgent
from rl_env import MetareasoningEnv

MODEL_PATH = "models/test-model.pth"

SEED = 1423
EPISODES = 10000
ENVIRONMENT = MetareasoningEnv()

INPUT_DIMENSION = ENVIRONMENT.observation_space.shape[0]
HIDDEN_DIMENSION = 64
OUTPUT_DIMENSION = ENVIRONMENT.action_space.n
LEARNING_RATE = 1e-3
SYNC_FREQUENCY = 5
EXPERIENCE_BUFFER_SIZE = 256
AGENT = RLAgent(seed=SEED, layer_sizes=[INPUT_DIMENSION, HIDDEN_DIMENSION, OUTPUT_DIMENSION], learning_rate=LEARNING_RATE, sync_frequency=SYNC_FREQUENCY, experience_buffer_size=EXPERIENCE_BUFFER_SIZE)

START_EPSILON = 1.0
END_EPSILON = 0.05

def main():
    losses_list, cumulative_rewards_list, episode_length_list, epsilon_list = [], [], [], []

    index = 0
    for _ in range(EXPERIENCE_BUFFER_SIZE):
        observation = ENVIRONMENT.reset()

        done = False
        while not done:
            action = AGENT.get_action(observation, ENVIRONMENT.action_space.n, 1.0)
            next_observation, reward, done, _ = ENVIRONMENT.step(action.item())
            AGENT.collect_experience([observation, action.item(), reward, next_observation])
            observation = next_observation

            index += 1

            if index > EXPERIENCE_BUFFER_SIZE:
                break

    epsilon = START_EPSILON

    index = 128
    for _ in tqdm(range(EPISODES)):
        observation, done, losses, episode_length, cumulative_reward = ENVIRONMENT.reset(), False, 0, 0, 0

        while not done:
            episode_length += 1

            action = AGENT.get_action(observation, ENVIRONMENT.action_space.n, epsilon)
            next_observation, reward, done, _ = ENVIRONMENT.step(action.item())
            AGENT.collect_experience([observation, action.item(), reward, next_observation])
            observation = next_observation

            cumulative_reward += reward

            index += 1

            if index > 128:
                index = 0
                for _ in range(4):
                    loss = AGENT.train(batch_size=16)
                    losses += loss

        if epsilon > END_EPSILON:
            epsilon -= (1 / 5000)

        losses_list.append(losses / episode_length)
        cumulative_rewards_list.append(cumulative_reward)
        episode_length_list.append(episode_length)
        epsilon_list.append(epsilon)

    AGENT.save_model(MODEL_PATH)


if __name__ == '__main__':
    main()
