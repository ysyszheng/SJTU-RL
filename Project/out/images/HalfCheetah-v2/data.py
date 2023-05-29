import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    rewards_ddpg = np.load('./out/datas/HalfCheetah-v2/ddpg.npy')
    rewards_sac = np.load('./out/datas/HalfCheetah-v2/sac.npy')

    # calculate average reward per 100 episodes, i.e. 0-99, 100-199, ...
    window_size = 200
    avg_rewards_ddpg = np.zeros(len(rewards_ddpg)//window_size)
    avg_rewards_sac = np.zeros(len(rewards_sac)//window_size)
    for i in range(0, len(rewards_ddpg), window_size):
        avg_rewards_ddpg[i//window_size] = np.mean(rewards_ddpg[i:i+window_size])
    for i in range(0, len(rewards_sac), window_size):
        avg_rewards_sac[i//window_size] = np.mean(rewards_sac[i:i+window_size])

    print('DDPG: ', avg_rewards_ddpg)
    print('SAC: ', avg_rewards_sac)