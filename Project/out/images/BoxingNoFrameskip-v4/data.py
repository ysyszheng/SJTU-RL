import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_dqn = np.load('./out/datas/BoxingNoFrameskip-v4/dqn.npy')
    rewards_ddqn = np.load('./out/datas/BoxingNoFrameskip-v4/ddqn.npy')
    rewards_d3qn = np.load('./out/datas/BoxingNoFrameskip-v4/d3qn.npy')

    # calculate average reward per 100 episodes, i.e. 0-99, 100-199, ...
    window_size = 600
    avg_rewards_dqn = np.zeros(len(rewards_dqn)//window_size)
    avg_rewards_ddqn = np.zeros(len(rewards_ddqn)//window_size)
    avg_rewards_d3qn = np.zeros(len(rewards_d3qn)//window_size)
    for i in range(0, len(rewards_dqn)-1, window_size):
        avg_rewards_dqn[i//window_size] = np.mean(rewards_dqn[i:i+window_size])
    for i in range(0, len(rewards_ddqn)-1, window_size):
        avg_rewards_ddqn[i//window_size] = np.mean(rewards_ddqn[i:i+window_size])
    for i in range(0, len(rewards_d3qn)-1, window_size):
        avg_rewards_d3qn[i//window_size] = np.mean(rewards_d3qn[i:i+window_size])

    print('DQN: ', avg_rewards_dqn)
    print('DDQN: ', avg_rewards_ddqn)
    print('D3QN: ', avg_rewards_d3qn)