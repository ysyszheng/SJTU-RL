from utils.replay_buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.max_action = max_action
    self.net = nn.Sequential(
      nn.Linear(state_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, action_dim),
      nn.Tanh()
    )

  def forward(self, state):
    return self.net(state) * self.max_action

  def act(self, state):
    with torch.no_grad():
      action = self.forward(state).cpu().data.numpy().flatten()
    return action
  
  def act_with_noise(self, state, noise):
    with torch.no_grad():
      action = self.forward(state).cpu().data.numpy().flatten()
    return (action + noise).clip(-self.max_action, self.max_action)
  
class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(state_dim + action_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, state, action):
    return self.net(torch.cat([state, action], 1))
  
class A3CWorker(object):
  def __init__(self, state_dim, action_dim, max_action, worker_id, 
               global_actor, global_critic, optimizer, device=torch.device("cpu")):
    self.worker_id = worker_id
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.global_actor = global_actor
    self.global_critic = global_critic
    self.optimizer = optimizer
    self.device = device

    # for actor_param in self.

  # def update(self, replay_buffer, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):

class A3C(object):
  def __init__(self, num_works, state_dim, action_dim, max_action, lr, device=torch.device("cpu")):
    self.num_works = num_works
    self.workers = [A3CWorker() for _ in range(num_works)]
    self.global_actor = Actor(state_dim, action_dim, max_action).to(device)
    self.global_critic = Critic(state_dim, action_dim).to(device)
    self.optimizer = torch.optim.Adam(self.actor.parameters(), lr)