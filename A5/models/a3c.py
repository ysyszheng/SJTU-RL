import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class ActorCritic(nn.Module):
    def __init__(self, state_dim, actor_dim, max_action):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, actor_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.max_action = max_action
        self.distribution = torch.distributions.Normal
        self.init_weight([self.actor, self.critic])

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        mu = policy * self.max_action
        sigma = torch.ones_like(mu) * 1e-3
        return (mu, sigma), value

    def init_weight(self, nets):
        for net in nets:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    nn.init.constant_(m.bias, 0)

    def select_action(self, state):
        self.training = False
        (mu, sigma), _ = self.forward(state)
        return self.distribution(mu.view(1, ).data, sigma.view(1, ).data).sample().numpy()

    # def loss(self, state, action, R, beta=0.01):
    #     self.train()
    #     (mu, sigma), value = self.forward(state)
    #     advantage = R - value
    #     d = self.distribution(mu, sigma)
    #     # entropy = d.entropy()
    #     entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(d.scale)
    #     critic_loss = advantage.pow(2)
    #     action_loss = -(d.log_prob(action) * advantage.detach() + beta * entropy)
    #     return (critic_loss + action_loss).mean()

    def save(self, path):
        torch.save(self.state_dict(), path + '_actor_critic.pt')

    def load(self, path):
        self.load_state_dict(torch.load(path + '_actor_critic.pt'))
