conda activate rl
# python3 -u ./train.py DQN
# python3 -u ./train.py DoubleDQN
# python3 -u ./train.py DuelingDQN
python3 -u ./test.py DQN
python3 -u ./test.py DoubleDQN
python3 -u ./test.py DuelingDQN