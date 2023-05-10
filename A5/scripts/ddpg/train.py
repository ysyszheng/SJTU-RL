import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.fix_seed import fix_seed
from utils.replay_buffer import ReplayBuffer
from models.ddpg import DDPG

class Trainer_DDPG(object):
  def __init__(self, config):
    # env
    self.config = config
    self.env = gym.make('Pendulum-v1', max_episode_steps=
                        self.config['max_steps'], g=self.config['g'])

    # fix seed
    fix_seed(self.config['seed'])
    
    # config
    state_dim = self.env.observation_space.shape[0]
    action_dim = self.env.action_space.shape[0]
    max_action = float(self.env.action_space.high[0])
    device = torch.device('cpu')
    lr = self.config['lr']
    gamma = self.config['gamma']
    tau = self.config['tau']
    batch_size = self.config['batch_size']
    memory_size = self.config['memory_size']

    # model
    self.agent = DDPG(state_dim, action_dim, max_action,\
                      lr, gamma, tau, batch_size, device)
    self.replay_buffer = ReplayBuffer(memory_size)

  def train(self):
    total_step = 0
    bar = tqdm(range(self.config['num_episodes']))
    r = []

    for episode in bar:
      state, _ = self.env.reset()
      # state, _ = self.env.reset(seed = self.config['seed'] + episode)
      episode_reward = 0

      while True:
        total_step += 1
        noise = np.random.normal(0, self.config['noise_std'], size=self.env.action_space.shape[0])
        action = self.agent.select_action_with_noise(state, noise)

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.add((state, action, next_state, reward, terminated))
        state = next_state
        episode_reward += reward
        
        if total_step > self.config['warmup_steps']:
          self.agent.update(self.replay_buffer, self.config['num_epochs'])
        if terminated or truncated:
          break
      
      bar.set_description('Episode: {}/{} | Episode Reward: {:.2f} | Terminal: {}'.
                          format(episode, self.config['num_episodes'], episode_reward, terminated))
      r.append(episode_reward)
    
    plt.figure()
    plt.plot(r)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(self.config['images_path'] + '/reward_train.png')

    # save r
    # np.save(self.config['reward_path'], r)

    self.agent.save(self.config['models_path'])
