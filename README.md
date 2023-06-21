SJTU CS3316 Reinforcement Learning
=====

Env
-----

```bash
pip install gymnasium
pip install 'gymnasium[classic-control]'
pip install 'gym[atari]'
pip install 'autorom[accept-rom-license]'
pip install 'gym[mujoco]'
pip install 'gym[mujoco_py]'
```

A1 - Iterative Policy Evaluation & Policy Iteration & Value Iteration
-----
Env: Grid World

```bash
python3 ./main.py
```

A2 - First-Visit MC & Every-Visit MC & TD(0)
-----
Env: Grid World

```bash
python3 ./main.py
```

A3 - Sarsa & Q-learning
-----
Env: Cliff Walking

```bash
python3 ./main.py
```

A4 - DQN & Double DQN & Dueling DQN
-----
Env: [Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

```bash
# Train
python3 ./train.py [Model Name: DQN or DoubleDQN or DuelingDQN]
# Test
python3 ./test.py [Model Name: DQN or DoubleDQN or DuelingDQN]
```

A5 - A3C & DDPG
-----
Env: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)

```bash
# Train
python3 ./run.py --mode train --model [Model Name: A3C or DDPG] --config [Config Path: ./config/a3c.yaml or ./config/ddpg.yaml]
# Test
python3 ./run.py --mode test --model [Model Name: A3C or DDPG] --config [Config Path: ./config/a3c.yaml or ./config/ddpg.yaml]
```

Project
-----

```bash
# Train
python3 ./run.py --env [Environment Name] --model [Model Name] --config [Config Path] --mode train
# Test
python3 ./run.py --env [Environment Name] --model [Model Name] --config [Config Path] --mode test
```
