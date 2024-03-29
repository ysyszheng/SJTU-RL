import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, c, h, w, action_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        w, h = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        # print(h,w) # TODO: delete
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        w, h = self.conv2d_size_calc(w, h, kernel_size=4, stride=2)
        # print(h,w) # TODO: delete
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        w, h = self.conv2d_size_calc(w, h, kernel_size=3, stride=1)
        # print(h,w) # TODO: delete
        self.fc1 = nn.Linear(w * h * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def conv2d_size_calc(self, w, h, kernel_size, stride):
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
class DDQN(object):
    def __init__(self, c, h, w, action_dim, lr, gamma, epsilon_min,
                 epsilon_decay, batch_size, device=torch.device('cpu')):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.0
        self.batch_size = batch_size
        self.device = device
        self.Q = Net(c, h, w, action_dim).to(device)
        self.Q_target = Net(c, h, w, action_dim).to(device)
        self.Q_target.eval()
        self.update_target()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def act_with_noise(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.action_dim, size=(1,)).item()
        else:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            # print(state.shape) # TODO: delete
            with torch.no_grad():
                return self.Q(state).argmax(1).item()
            
    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.Q(state).argmax(1).item()
            
    def update(self, replay_buffer, iterations):
      for _ in range(iterations):
        # Sample replay buffer
        s, a, s_, r, d = replay_buffer.sample(self.batch_size)
        state = torch.from_numpy(np.array(s)).float().to(self.device)
        action = torch.from_numpy(a).long().to(self.device)
        next_state = torch.from_numpy(np.array(s_)).float().to(self.device)
        reward = torch.from_numpy(r).float().to(self.device)
        done = torch.from_numpy(1-d).float().to(self.device)

        # update using double DQN algorithm
        current_Q = self.Q(state).gather(1, action.unsqueeze(1))
        max_action = self.Q(next_state).max(1)[1].unsqueeze(1)
        target_Q = reward + (done * self.gamma * self.Q_target(next_state)
                             .gather(1, max_action)).detach()

        # Compute Q loss
        loss = self.loss(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def set_eval(self):
        self.Q.eval()
        self.Q_target.eval()

    def save(self, path):
        # torch.save(self.Q.to('cpu').state_dict(), path)
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.Q_target.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
