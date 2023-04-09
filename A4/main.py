import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

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

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            act_values = self.model(state)
            return np.argmax(act_values[0]).detach().numpy()

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
        # self.loss.append(loss.item())

        self.model.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('MountainCar-v0', render_mode='human').env # avoid truncation
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = BATCH_SIZE

    for e in range(EPISODES):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                agent.loss.append(time)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % TARGET_UPDATE == 0:
            agent.update_target_model()

    # agent.save("./save/mountaincar-dqn.h5")
    plt.plot([i+1 for i in range(0, len(agent.loss), 2)], agent.loss[::2])
    plt.ylabel('Episode Length')
    plt.xlabel('Episode')
    plt.show()
