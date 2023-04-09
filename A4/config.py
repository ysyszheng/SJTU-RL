import torch

config = {
  'episode': 2000,
  'max_steps': 500,
  'batch_size': 32,
  'gamma': 0.99,
  'epsilon': 1.0,
  'epsilon_decay': 0.999,
  'epsilon_min': 0.01,
  'learning_rate': 0.001,
  'target_update': 20,
  'memory_size': 512
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'