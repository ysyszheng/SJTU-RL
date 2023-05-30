import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
from tqdm import tqdm
from utils.fix_seed import fix_seed
from utils.replay_buffer import ReplayBuffer
from models.ddqn import DDQN
import os

class Trainer(object):
    def __init__(self, config):
        # create env
        self.config = config
        # self.env = gym.make(config['env'], max_episode_steps=self.config['max_steps'])
        self.env = gym.make(config['env'])
        self.env = AtariPreprocessing(self.env, scale_obs=True) # TODO: need FrameStack?
        self.env = FrameStack(self.env, num_stack=4)
        fix_seed(self.config['seed'])

        # params
        self.c = self.env.observation_space.shape[0]
        self.h = self.env.observation_space.shape[1]
        self.w = self.env.observation_space.shape[2]
        # print(self.env.observation_space.shape)
        # self.target_h = self.config['target_h']
        # self.target_w = self.config['target_w']
        # self.target_c = 1
        action_dim = self.env.action_space.n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = self.config['lr']
        gamma = self.config['gamma']
        batch_size = self.config['batch_size']
        memory_size = self.config['memory_size']
        
        # model
        self.agent = DDQN(self.c, self.h, self.w, action_dim, lr, gamma, 
                         config['epsilon_min'], config['epsilon_decay'], batch_size, device)
        self.replay_buffer = ReplayBuffer(memory_size)

        # save path
        self.data_dir = "./out/datas/" + self.config['env']
        self.model_dir = "./out/models/" + self.config['env']
        if os.path.exists(self.data_dir) == False:
            os.makedirs(self.data_dir)
        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)

    def process(self, state):
        # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.resize(state, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        state = np.expand_dims(state, axis=0)
        # print(state) # TODO: delete
        return state

    def train(self):
        total_step = 0
        bar = tqdm(range(self.config['num_episodes']))
        r = []

        for episode in bar:
            state, _ = self.env.reset()
            # state = self.process(state)
            # state, _ = self.env.reset(seed = self.config['seed'] + episode)
            episode_reward = 0

            while True:
                total_step += 1
                action = self.agent.act_with_noise(state)
                # print(action) # TODO: delete
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                # print(reward) # TODO: delete
                # next_state = self.process(next_state)
                self.replay_buffer.add(
                    (state, action, next_state, reward, terminated or truncated))
                state = next_state
                episode_reward += reward

                if total_step > self.config['warmup_steps'] and total_step % self.config['update_freq'] == 0:
                    self.agent.update(self.replay_buffer,
                                      self.config['num_epochs'])
                    # print("update")
                if total_step > self.config['warmup_steps'] and total_step % self.config['target_update'] == 0:
                    self.agent.update_target()
                    # print("update target")
                if terminated or truncated:
                    break

            bar.set_description('Episode: {}/{} | Episode Reward: {:.2f} | Truncated: {} | Terminated: {}'.
                                format(episode+1, self.config['num_episodes'], episode_reward, truncated, terminated))
            r.append(episode_reward)

            # if episode % self.config['save_interval'] == 0:
            #     np.save(self.data_dir + f"/ddqn_{episode}.npy", r)
            #     self.agent.save(self.model_dir + f"/ddqn_{episode}.pt")

        # save rewards
        np.save(self.data_dir + "/ddqn.npy", r)
        self.agent.save(self.model_dir + "/ddqn.pt")
