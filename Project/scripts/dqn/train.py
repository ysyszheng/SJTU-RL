import gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.fix_seed import fix_seed
from utils.replay_buffer import ReplayBuffer
from models.dqn import DQN
import cv2

class Trainer(object):
    def __init__(self, config):
        # create env
        self.config = config
        # self.env = gym.make(config['env'], max_episode_steps=self.config['max_steps'])
        self.env = gym.make(config['env'])
        fix_seed(self.config['seed'])

        # params
        self.h = self.env.observation_space.shape[0]
        self.w = self.env.observation_space.shape[1]
        self.c = self.env.observation_space.shape[2]
        self.target_h = self.config['target_h']
        self.target_w = self.config['target_w']
        self.target_c = 1
        action_dim = self.env.action_space.n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = self.config['lr']
        gamma = self.config['gamma']
        epsilon = self.config['epsilon']
        batch_size = self.config['batch_size']
        memory_size = self.config['memory_size']
        
        # model
        self.agent = DQN(self.target_c, self.target_h, self.target_w, action_dim, lr, gamma, epsilon, batch_size, device)
        self.replay_buffer = ReplayBuffer(memory_size)

    def process(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (self.target_h, self.target_w), interpolation=cv2.INTER_AREA)
        state = np.expand_dims(state, axis=0)
        # state = np.transpose(state, (2, 0, 1))
        # state = torch.from_numpy(state).float()
        return state

    def train(self):
        total_step = 0
        bar = tqdm(range(self.config['num_episodes']))
        r = []

        for episode in bar:
            state, _ = self.env.reset()
            state = self.process(state)
            # state, _ = self.env.reset(seed = self.config['seed'] + episode)
            episode_reward = 0

            while True:
                total_step += 1
                action = self.agent.act_with_noise(state)

                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                next_state = self.process(next_state)
                self.replay_buffer.add(
                    (state, action, next_state, reward, terminated))
                state = next_state
                episode_reward += reward

                if total_step > self.config['warmup_steps']:
                    self.agent.update(self.replay_buffer,
                                      self.config['num_epochs'])
                if total_step % self.config['target_update'] == 0:
                    self.agent.update_target()
                if terminated or truncated:
                    break

            bar.set_description('Episode: {}/{} | Episode Reward: {:.2f} | Truncated: {} | Terminated: {}'.
                                format(episode+1, self.config['num_episodes'], episode_reward, truncated, terminated))
            r.append(episode_reward)

        # save rewards
        np.save("./out/datas/" + self.config['env'] + "/dqn.npy", r)
        self.agent.save("./out/models/" + self.config['env'] + "/dqn")
