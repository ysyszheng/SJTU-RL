import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_dqn = np.load('./out/datas/BoxingNoFrameskip-v4/dqn_3000.npy')
    # rewards_ddqn = np.load('./out/datas/BoxingNoFrameskip-v4/ddqn.npy')
    # rewards_d3qn = np.load('./out/datas/BoxingNoFrameskip-v4/d3qn.npy')

    window_size = 5
    rewards_dqn = np.convolve(rewards_dqn, np.ones(window_size)/window_size, mode='valid')
    # rewards_ddqn = np.convolve(rewards_ddqn, np.ones(window_size)/window_size, mode='valid')
    # rewards_d3qn = np.convolve(rewards_d3qn, np.ones(window_size)/window_size, mode='valid')

    plt.plot(rewards_dqn, label='DQN')
    # plt.plot(rewards_ddqn, label='DDQN')
    # plt.plot(rewards_d3qn, label='D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./out/images/BoxingNoFrameskip-v4/BoxingNoFrameskip-v4.png')
    