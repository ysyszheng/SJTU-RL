import numpy as np
import gymnasium as gym
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
    bar = tqdm(range(EPISODES))

    for e in bar: # episode
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        terminated, truncated = False, False
        score, step = 0, 0

        while not terminated and not truncated: # step, max_step = 200
            step += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = reward if not terminated else 0
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            if step % TARGET_UPDATE == 0:
                agent.update_target_model()
        
        agent.score.append(score)
        bar.set_description("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, score, agent.epsilon))

    agent.savemodel("./models/dqn.pt")
    agent.saveloss("./images/dqn-loss.png")
    agent.savescore("./images/dqn-reward.png")
