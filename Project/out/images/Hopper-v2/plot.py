import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_ddpg = np.load('./out/datas/Hopper-v2/ddpg.npy')
    # rewards_sac = np.load('./out/datas/Hopper-v2/sac.npy')
    plt.plot(rewards_ddpg, label='DDPG')
    # plt.plot(rewards_sac, label='SAC')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./out/images/Hopper-v2/Hopper-v2.png')
    