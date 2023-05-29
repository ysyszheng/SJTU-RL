import torch
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from models.ddqn import DDQN
from utils.fix_seed import fix_seed
from utils.wrappers import ClipReward
import numpy as np

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
        self.env = AtariPreprocessing(self.env, scale_obs=True)
        self.env = FrameStack(self.env, num_stack=4)
        self.env = ClipReward(self.env, -1, 1)
        fix_seed(self.config['seed'] + 666)

        # params
        self.c = self.env.observation_space.shape[0]
        self.h = self.env.observation_space.shape[1]
        self.w = self.env.observation_space.shape[2]
        action_dim = self.env.action_space.n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = self.config['lr']
        gamma = self.config['gamma']
        batch_size = self.config['batch_size']
        
        # model
        self.agent = DDQN(self.c, self.h, self.w, action_dim, lr, gamma, 
                          config['epsilon_min'], config['epsilon_decay'], batch_size, device)
        self.agent.load("./out/models/" + self.config['env'] + "/ddqn.pt")

    def process(self, state):
        # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.resize(state, (self.w, self.h), interpolation=cv2.INTER_AREA)
        state = np.expand_dims(state, axis=0)
        # state = np.transpose(state, (2, 0, 1))
        # state = torch.from_numpy(state).float()
        return state

    def test(self):
        self.agent.set_eval()
        r = []

        for _ in range(self.config['test_episodes']):
            state, _ = self.env.reset()
            # state = self.process(state)
            episode_reward = 0
            while True:
                action = self.agent.act(state)
                # print(action)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                # state = self.process(next_state)
                state = next_state
                episode_reward += reward
                if terminated or truncated:
                    break
            r.append(episode_reward)
            print(f'Episode Reward: {episode_reward}, Truncated: {truncated}, terminated: {terminated}')
