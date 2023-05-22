import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)
    # rewards_dqn = np.load('./out/datas/BoxingNoFrameskip-v4/dqn.npy')
    rewards_ddqn = np.load('./out/datas/BoxingNoFrameskip-v4/ddqn.npy')
    rewards_d3qn = np.load('./out/datas/BoxingNoFrameskip-v4/d3qn.npy')
    # plt.plot(rewards_dqn, label='DQN')
    plt.plot(rewards_ddqn, label='DDQN')
    plt.plot(rewards_d3qn, label='D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('./out/images/BoxingNoFrameskip-v4/BoxingNoFrameskip-v4.png')
    