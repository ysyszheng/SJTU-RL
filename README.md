Env
-----
```bash
pip install gymnasium
pip install "gymnasium[classic-control]"
```

A4 - Mountain Car
-----
[Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

```bash
# Train
python3 ./train.py [Model Name: DQN/DoubleDQN/DuelingDQN]
# Test
python3 ./test.py [Model Name: DQN/DoubleDQN/DuelingDQN]
```

A5 - Pendulum
-----
[Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)

```bash
# Train
python3 ./run.py --mode train --model [Model Name: A3C or DDPG] --config [Config Path: ./config/a3c.yaml or ./config/ddpg.yaml]
# Test
python3 ./run.py --mode test --model [Model Name: A3C or DDPG] --config [Config Path: ./config/a3c.yaml or ./config/ddpg.yaml]
```
