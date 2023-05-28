import gym
import torch
import numpy as np
from tqdm import tqdm
from utils.fix_seed import fix_seed
from utils.replay_buffer import ReplayBuffer
from models.sac import SAC


class Trainer(object):
    def __init__(self, config):
        # create env
        self.config = config
        # self.env = gym.make(config['env'], max_episode_steps=self.config['max_steps'])
        self.env = gym.make(config['env'])
        fix_seed(self.config['seed'])

        # params
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        device = torch.device('cpu')
        lr = self.config['lr']
        self.num_epochs = self.config['num_epochs']
        self.batch_size = self.config['batch_size']
        self.policy_noise = self.config['policy_noise']
        self.noise_clip = self.config['noise_clip']
        self.policy_freq = self.config['policy_freq']
        gamma = self.config['gamma']
        tau = self.config['tau']
        alpha = self.config['alpha']
        memory_size = self.config['memory_size']

        # model
        self.agent = SAC(state_dim, action_dim, max_action, 
                         gamma, tau, alpha, lr, device)
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
                action = self.agent.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                self.replay_buffer.add(
                    (state, action, next_state, reward, terminated))
                state = next_state
                episode_reward += reward if not truncated else 0 # ignore truncated

                if total_step > self.config['warmup_steps']:
                    self.agent.train(self.replay_buffer, self.num_epochs, self.batch_size) 
                if terminated or truncated:
                    break

            bar.set_description('Episode: {}/{} | Episode Reward: {:.2f} | Truncated: {} | Terminated: {}'.
                                format(episode+1, self.config['num_episodes'], episode_reward, truncated, terminated))
            r.append(episode_reward)

        # save rewards
        np.save("./out/datas/" + self.config['env'] + "/sac.npy", r)

        self.agent.save("./out/models/" + self.config['env'] + "/sac")
