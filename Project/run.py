import argparse
import yaml

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default=None, help='environment name: \nPongNoFrameskip-v4, BreakoutNoFrameskip-v4, Hopper-v2, Ant-v2')
  parser.add_argument('--mode', type=str, default='train', help='train or test')
  parser.add_argument('--model', type=str, default=None, help='model name: DQN, DDQN, D3QN, DDPG, SAC')
  parser.add_argument('--config', type=str, default=None, help='config file path')
  args = parser.parse_args()

  if args.config is not None:
    with open(args.config, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
  else:
    config = {}

  if args.env is not None:
    config['env'] = args.env

  if args.model == 'DQN':
    if args.mode == 'train':
      from scripts.dqn.train import Trainer
      trainer = Trainer(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.dqn.test import Tester
      tester = Tester(config)
      tester.test()
    else:
      raise NotImplementedError
    
  elif args.model == 'DDQN':
    if args.mode == 'train':
      from scripts.ddqn.train import Trainer
      trainer = Trainer(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.ddqn.test import Tester
      tester = Tester(config)
      tester.test()

  elif args.model == 'D3QN':
    if args.mode == 'train':
      from scripts.d3qn.train import Trainer
      trainer = Trainer(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.d3qn.test import Tester
      tester = Tester(config)
      tester.test()

  elif args.model == 'DDPG':
    if args.mode == 'train':
      from scripts.ddpg.train import Trainer
      trainer = Trainer(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.ddpg.test import Tester
      tester = Tester(config)
      tester.test()
    else:
      raise NotImplementedError

  elif args.model == 'SAC':
    if args.mode == 'train':
      from scripts.sac.train import Trainer
      trainer = Trainer(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.sac.test import Tester
      tester = Tester(config)
      tester.test()

  else:
    raise NotImplementedError
