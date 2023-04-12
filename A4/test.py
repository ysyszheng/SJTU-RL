import sys
import gymnasium as gym
from model import DQNAgent, DoubleDQNAgent, DuelingDQNAgent

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='human')
    # env = gym.make('MountainCar-v0')
    state, info = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if len(sys.argv) != 2:
        print('Usage: python3 test.py DQN/DoubleDQN/DuelingDQN')
        exit(1)
    if sys.argv[1] == 'DQN':
      agent = DQNAgent(state_size, action_size)
      agent.load_model('./models/dqn.pt')
      score_path = './images/dqn-reward-test.png'
    elif sys.argv[1] == 'DoubleDQN':
      agent = DoubleDQNAgent(state_size, action_size)
      agent.load_model('./models/double-dqn.pt')
      score_path = './images/double-dqn-reward-test.png'
    elif sys.argv[1] == 'DuelingDQN':
      agent = DuelingDQNAgent(state_size, action_size)
      agent.load_model('./models/dueling-dqn.pt')
      score_path = './images/dueling-dqn-reward-test.png'
    else:
      print('Usage: python3 test.py DQN/DoubleDQN/DuelingDQN')
      exit(1)

    for i in range(100):
      score = 0
      terminated, truncated = False, False

      while not terminated and not truncated:
        action = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        reward = reward if not terminated else 0
        score += reward

      agent.score.append(score)
      print(f'episode: {i}/100, reward: {score}, average reward: {sum(agent.score)/len(agent.score)}')
      state, info = env.reset()

    env.close()
    # agent.save_score(score_path)
