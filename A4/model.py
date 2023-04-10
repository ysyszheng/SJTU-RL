import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import config
import matplotlib.pyplot as plt

GAMMA = config['gamma']
EPSILON = config['epsilon']
EPSILON_DECAY = config['epsilon_decay']
EPSILON_MIN = config['epsilon_min']
LEARNING_RATE = config['learning_rate']
MEMORY_SIZE = config['memory_size']
EPOCH = config['epoch']

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.apply(self.init_weights)
        self.update_target_model()
        self.loss = []
        self.score = []

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )
        return model
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1/np.sqrt(m.in_features))
            m.bias.data.fill_(0.01)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.model.training and (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        act_values = self.model(torch.from_numpy(state).float()).detach()
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        for _ in range(EPOCH):
            minibatch = random.sample(self.memory, batch_size)
            states = torch.from_numpy(np.vstack([i[0] for i in minibatch])).float()
            actions = torch.from_numpy(np.vstack([i[1] for i in minibatch])).long()
            rewards = torch.from_numpy(np.vstack([i[2] for i in minibatch])).float()
            next_states = torch.from_numpy(np.vstack([i[3] for i in minibatch])).float()
            dones = torch.from_numpy(np.vstack([i[4] for i in minibatch]).astype(np.uint8)).float()

            Q_expected = self.model(states).gather(1, actions)
            Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

            loss = F.mse_loss(Q_expected, Q_targets)
            self.loss.append(loss.item())

            self.model.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def savemodel(self, path):
        torch.save(self.model.state_dict(), path)

    def saveloss(self, path):
        plt.figure()
        plt.plot(self.loss)
        plt.ylabel('Loss')
        plt.xlabel('Step')
        plt.savefig(path, dpi=200)

    def savescore(self, path):
        avg_score = []
        for i in range(len(self.score)):
            avg_score.append(np.mean(self.score[max(0,i-100):(i+1)]))

        plt.figure()
        plt.plot(self.score)
        plt.plot(avg_score, '-.')
        plt.plot([0, len(self.score)], [-200, -200], 'r--')
        plt.plot([0, len(self.score)], [-110, -110], 'g--')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.legend(['Reward', 'Average Reward', 'Truncated (-200)', 'Solved (-110)'], loc='lower right')
        plt.savefig(path, dpi=200)

# Dueling DQN Agent
# Q(s, a) = V(s) + A(s, a)
class DuelingDQNAgent(DQNAgent):
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Linear(64, self.action_size),
        )
        return model

    def act(self, state):
        if self.model.training and (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        act_values = self.model(torch.from_numpy(state).float()).detach()
        return torch.argmax(act_values[0] + act_values[1] - act_values[1].mean()).item()