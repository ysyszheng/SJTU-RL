import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import config
from model import DQNAgent

EPISODES = config['episode']
BATCH_SIZE = config['batch_size']
TARGET_UPDATE = config['target_update']

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    bar = tqdm(range(EPISODES))

    for e in bar: # episode
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        step = 0

        while True: # step
            step += 1
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            reward = reward if not done else 0
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            if step % TARGET_UPDATE == 0:
                agent.update_target_model()
            if done or truncated:
                break

        agent.score.append(score)
        bar.set_description("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, score, agent.epsilon))

    agent.save("./models/dqn.pt")
    plt.plot([i+1 for i in range(len(agent.score))], agent.score)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig("./images/dqn.png", dpi=200)
