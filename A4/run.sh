conda activate rl
cd /Users/yusen/Documents/Study/CS3316-强化学习/A/A4
python3 -u ./train.py DQN
python3 -u ./train.py DoubleDQN
python3 -u ./train.py DuelingDQN
python3 -u ./test.py DQN
python3 -u ./test.py DoubleDQN
python3 -u ./test.py DuelingDQN