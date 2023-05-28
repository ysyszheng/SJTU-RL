import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_ddpg = np.load('./out/datas/Ant-v2/ddpg.npy')
    rewards_sac = np.load('./out/datas/Ant-v2/sac.npy')

    window_size = 5
    rewards_ddpg = np.convolve(rewards_ddpg, np.ones(window_size)/window_size, mode='valid')
    rewards_sac = np.convolve(rewards_sac, np.ones(window_size)/window_size, mode='valid')

    plt.plot(rewards_ddpg, label='DDPG')
    plt.plot(rewards_sac, label='SAC')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./out/images/Ant-v2/Ant-v2.png')
    