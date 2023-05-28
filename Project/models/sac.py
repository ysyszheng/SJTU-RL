import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.l4 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(self.weights_init)

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
        action = self.max_action * action
        log_prob = normal.log_prob(z) - torch.log(
            self.max_action*(1 - (action/self.max_action).pow(2) + 1e-6)) # TODO: check this
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, z, mean, log_std
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
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

        self.apply(self.weights_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, gamma, tau, alpha, lr=3e-4, device=torch.device('cpu')):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.alpha = alpha
        self.H_bar = torch.tensor([-action_dim]).to(self.device).float()
        self.log_alpha = torch.tensor([1.0], requires_grad=True, device=self.device).float()
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _, _, _, _ = self.actor.forward(state)
        return action.detach().cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, num_epochs=1, batch_size=64):
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
                # noise = torch.FloatTensor(action).data.normal_(0, policy_noise).to(self.device)
                # noise = noise.clamp(-noise_clip, noise_clip)
                next_action, next_log_pi, _, _, _ = self.actor(next_state)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q1, target_Q2 = target_Q1.detach(), target_Q2.detach()
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + self.gamma * done * (target_Q - self.alpha * next_log_pi)

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
            action, log_pi, _, _, _ = self.actor(state)
            Q = torch.min(self.critic.Q1(state, action), self.critic.Q2(state, action))
            actor_loss = (self.alpha * log_pi - Q).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha
            loss_log_alpha = self.log_alpha * (-log_pi.detach() - self.H_bar).mean()
            self.log_alpha_optimizer.zero_grad()
            loss_log_alpha.backward()
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            # Update the frozen target models
            self.update_target()

    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.actor.state_dict(), filename + "_actor.pt")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()