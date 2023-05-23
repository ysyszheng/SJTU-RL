import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Soft Actor-Critic (SAC)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-2, log_std_max=-20):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.l4 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.l3(a)
        log_std = self.l4(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(self.max_action*(1 - action).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action = self.max_action * action
        return action, log_prob, z, mean, log_std
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, device=torch.device('cpu')):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _, _, _, _ = self.actor.forward(state)
        return action.detach().cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, num_epochs=1, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5):
        for _ in range(num_epochs):
            # Sample replay buffer 
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(1 - done).to(self.device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise 
                noise = torch.FloatTensor(action).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action, next_log_pi, _, _, _ = self.actor(next_state)
                next_action = (next_action + noise).clamp(-self.actor.max_action, self.actor.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic(next_state, next_action)
                target_Q1, target_Q2 = target_Q1.detach(), target_Q2.detach()
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * (target_Q - next_log_pi))

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # # Delayed policy updates
            # if random.randint(1, 100) % policy_freq == 0:

            # Compute actor loss
            _, log_pi, _, _, _ = self.actor(state)
            actor_loss = (log_pi.detach() - current_Q1).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        torch.load(self.critic.state_dict(), filename + "_critic")
        torch.load(self.actor.state_dict(), filename + "_actor")
