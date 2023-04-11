import sys
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from config import config
from model import DQNAgent, DoubleDQNAgent, DuelingDQNAgent

EPISODES = config['episode']
MAX_STEPS = config['max_steps']
BATCH_SIZE = config['batch_size']
TARGET_UPDATE = config['target_update']

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    args = sys.argv
    if len(args) != 2:
        print('Usage: python3 train.py DQN/DoubleDQN/DuelingDQN')
        exit(1)
    if args[1] == 'DQN':
        agent = DQNAgent(state_size, action_size)
        model_path = './models/dqn.pt'
        loss_path = './images/dqn-loss.png'
        score_path = './images/dqn-score.png'
        score_100_path = './images/dqn-last-100-reward.png'
    elif args[1] == 'DoubleDQN':
        agent = DoubleDQNAgent(state_size, action_size)
        model_path = './models/double-dqn.pt'
        loss_path = './images/double-dqn-loss.png'
        score_path = './images/double-dqn-score.png'
        score_100_path = './images/double-dqn-last-100-reward.png'
    elif args[1] == 'DuelingDQN':
        agent = DuelingDQNAgent(state_size, action_size)
        model_path = './models/dueling-dqn.pt'
        loss_path = './images/dueling-dqn-loss.png'
        score_path = './images/dueling-dqn-score.png'
        score_100_path = './images/dueling-dqn-last-100-reward.png'
    else:
        print('Usage: python3 train.py DQN/DoubleDQN/DuelingDQN')
        exit(1)

    bar = tqdm(range(EPISODES))
    for e in bar: # episode
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        terminated, truncated = False, False
        score, step = 0, 0

        # while not terminated and not truncated: # step
        while not terminated and step <= MAX_STEPS: # step
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
                    .format(e+1, EPISODES, score, agent.epsilon))
    
    env.close()
    agent.save_model(model_path)
    agent.save_loss(loss_path)
    agent.save_score(score_path)
    agent.save_last_100_scores(score_100_path)
