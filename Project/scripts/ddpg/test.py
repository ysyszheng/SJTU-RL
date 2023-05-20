import torch
import gym
from models.ddpg import DDPG
from utils.fix_seed import fix_seed


class Tester(object):
    def __init__(self, config):
        # create env
        self.config = config
        if self.config['render']:
            # self.env = gym.make(config['env'], render_mode="human", max_episode_steps=config['max_steps'])
            self.env = gym.make(config['env'], render_mode="human")
        else:
            # self.env = gym.make(config['env'], max_episode_steps=config['max_steps'])
            self.env = gym.make(config['env'])
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
        self.agent = DDPG(state_dim, action_dim, max_action,
                          lr, gamma, tau, batch_size, device)
        self.agent.load("./out/models/" + self.config['env'] + "/ddpg")

    def test(self):
        self.agent.set_eval()
        r = []

        for _ in range(self.config['test_episodes']):
            state, _ = self.env.reset()
            episode_reward = 0
            while True:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                state = next_state
                episode_reward += reward
                if terminated or truncated:
                    break
            r.append(episode_reward)
            print(f'Episode Reward: {episode_reward}, Truncated: {truncated}, terminated: {terminated}')
