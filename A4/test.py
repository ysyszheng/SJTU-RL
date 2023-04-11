import gymnasium as gym
from model import DQNAgent

# https://github.com/openai/gym/wiki/MountainCar-v0:
# MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='human')
    state, info = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load_model("./models/dqn.pt")
    agent.load_model("./models/dueling-dqn.pt")
    # agent.epsilon = 0.0

    for _ in range(100):
      score = 0
      terminated, truncated = False, False

      while not terminated and not truncated:
        action = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        reward = reward if not terminated else 0
        score += reward

      agent.score.append(score)
      print(f'reward: {score}, average reward: {sum(agent.score)/len(agent.score)}')
      state, info = env.reset()

    env.close()
