import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class DQN(object):
    def __init__(self, state_dim, action_dim, lr, gamma, 
                 epsilon, batch_size, device=torch.device('cpu')):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device
        self.Q = Net(state_dim, action_dim).to(device)
        self.Q_target = Net(state_dim, action_dim).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def act_with_noise(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.action_dim, size=(1,))
        else:
            state = torch.from_numpy(np.array(state) / 255.0).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.Q(state).argmax(1)
            
    def act(self, state):
        state = torch.from_numpy(np.array(state) / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.Q(state).argmax(1)
            
    def update(self, replay_buffer, iterations):
      for _ in range(iterations):
        # Sample replay buffer
        s, a, s_, r, d = replay_buffer.sample(self.batch_size)
        state = torch.from_numpy(np.array(s) / 255.0).float().to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        next_state = torch.from_numpy(np.array(s_) / 255.0).float().to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(1 - d).to(self.device)

        # Compute the target Q value
        target_Q = self.Q_target(next_state).max(1)[0].unsqueeze(1)
        target_Q = reward + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action.long())

        # Compute Q loss
        loss = self.loss(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, path):
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path))
        self.Q_target.load_state_dict(torch.load(path))
