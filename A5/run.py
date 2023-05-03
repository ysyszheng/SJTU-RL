import argparse
import yaml

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='train or test')
  parser.add_argument('--model', type=str, default=None, help='model name, A3C or DDPG')
  parser.add_argument('--config', type=str, default=None, help='config file path')
  args = parser.parse_args()

  if args.config is not None:
    with open(args.config, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
  else:
    config = {}

  if args.model == 'A3C':
    if args.mode == 'train':
      from scripts.a3c.train import Trainer_A3C
      trainer = Trainer_A3C(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.a3c.test import Tester_A3C
      tester = Tester_A3C(config)
      tester.test()
    else:
      raise NotImplementedError

  elif args.model == 'DDPG':
    if args.mode == 'train':
      from scripts.ddpg.train import Trainer_DDPG
      trainer = Trainer_DDPG(config)
      trainer.train()
    elif args.mode == 'test':
      from scripts.ddpg.test import Tester_DDPG
      tester = Tester_DDPG(config)
      tester.test()
    else:
      raise NotImplementedError

  else:
    raise NotImplementedError
