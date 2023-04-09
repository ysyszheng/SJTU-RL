import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import config

GAMMA = config['gamma']
EPSILON = config['epsilon']
EPSILON_DECAY = config['epsilon_decay']
EPSILON_MIN = config['epsilon_min']
LEARNING_RATE = config['learning_rate']
MEMORY_SIZE = config['memory_size']

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

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(torch.from_numpy(state).float())
        return np.argmax(act_values[0].numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state = torch.from_numpy(np.vstack([i[0] for i in minibatch])).float()
        action = torch.from_numpy(np.vstack([i[1] for i in minibatch])).long()
        reward = torch.from_numpy(np.vstack([i[2] for i in minibatch])).float()
        next_state = torch.from_numpy(np.vstack([i[3] for i in minibatch])).float()
        done = torch.from_numpy(np.vstack([i[4] for i in minibatch]).astype(np.uint8)).float()

        Q_value = self.model(state).gather(1, action)
        Q_next = self.target_model(next_state).detach().max(1)[0].unsqueeze(1)
        Q_target = reward + (self.gamma * Q_next * (1 - done))
        loss = F.mse_loss(Q_value, Q_target)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss.append(loss.item())

        self.model.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

# Dueling DQN Agent
