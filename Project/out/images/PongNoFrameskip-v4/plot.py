import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_dqn = np.load('./out/datas/PongNoFrameskip-v4/dqn.npy')
    rewards_ddqn = np.load('./out/datas/PongNoFrameskip-v4/ddqn.npy')
    rewards_d3qn = np.load('./out/datas/PongNoFrameskip-v4/d3qn.npy')

    window_size = 5
    rewards_dqn_avg = np.convolve(rewards_dqn, np.ones(window_size)/window_size, mode='valid')
    rewards_ddqn_avg = np.convolve(rewards_ddqn, np.ones(window_size)/window_size, mode='valid')
    rewards_d3qn_avg = np.convolve(rewards_d3qn, np.ones(window_size)/window_size, mode='valid')

    plt.plot(rewards_dqn_avg, label='DQN')
    plt.plot(rewards_ddqn_avg, label='DDQN')
    plt.plot(rewards_d3qn_avg, label='D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./out/images/PongNoFrameskip-v4/PongNoFrameskip-v4.png')