import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from models.ddpg import DDPG
from utils.fix_seed import fix_seed

class Tester_DDPG(object):
    def __init__(self, config):
        # env
        self.config = config
        
        if self.config['render']:
            self.env = gym.make('Pendulum-v1', render_mode="human", g=self.config['g'])
        else:
            self.env = gym.make('Pendulum-v1', g=self.config['g'])

        # fix seed
        fix_seed(self.config['seed'] + 666)

        # config
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        device = torch.device('cpu')
        lr = self.config['lr']
        gamma = self.config['gamma']
        tau = self.config['tau']
        batch_size = self.config['batch_size']

        # model
        self.agent = DDPG(state_dim, action_dim, max_action, \
                          lr, gamma, tau, batch_size, device)
        self.agent.load(self.config['models_path'])
    
    def test(self):
        self.agent.set_eval()
        r = []

        for _ in range(self.config['test_episodes']):
            state, _ = self.env.reset()
            episode_reward = 0
            while True:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                if terminated or truncated:
                    break
            r.append(episode_reward)
            print('Episode Reward: {:.2f}'.format(episode_reward))

        plt.figure()
        plt.plot(r)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.config['images_path'] + '/reward_test.png')
