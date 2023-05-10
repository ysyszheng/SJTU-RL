import gymnasium as gym
import torch
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from utils.fix_seed import fix_seed
from utils.shared_adam import SharedAdam
from models.a3c import ActorCritic
import os
os.environ["OMP_NUM_THREADS"] = "1"

class Trainer_A3C(object):
    def __init__(self, config):
        # env
        self.config = config
        self.env = gym.make(
            'Pendulum-v1', max_episode_steps=self.config['max_steps'], g=self.config['g'])

        # config
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.lr = self.config['lr']
        self.device = torch.device('cpu')
        self.gamma = self.config['gamma']
        self.beta = self.config['beta']
        self.lambd = self.config['lambda']
        self.value_coef = self.config['value_coef']
        self.T_max = self.config['T_max']
        self.t_max = self.config['t_max']
        self.num_workers = self.config['num_workers']
        self.max_grad_norm = self.config['max_grad_norm']

        # mp
        self.total_step = mp.Value('i', 0)
        # self.r_list = mp.Manager().list()
        self.lock = mp.Lock()

        # model
        self.global_model = ActorCritic(
            self.state_dim, self.action_dim, self.max_action)
        self.global_model.share_memory()
        self.optimizer = SharedAdam(self.global_model.parameters(), lr=self.lr)
        self.optimizer.share_memory()

    def train(self):
        processes = []
        for rank in range(self.num_workers):
            p = mp.Process(target=self.train_worker, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        self.global_model.save(self.config['models_path'])
        # plt.figure()
        # plt.plot(self.r_list)
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.savefig(self.config['images_path'] + '/reward_train_global.png')

    def train_worker(self, rank):
        fix_seed(self.config['seed'] + rank)
        reward_list = []

        worker_model = ActorCritic(
            self.state_dim, self.action_dim, self.max_action)
        worker_model.train()

        while self.total_step.value < self.T_max:
            state, _ = self.env.reset()
            worker_model.load_state_dict(self.global_model.state_dict())
            values, log_probs, rewards, entropies = [], [], [], []
            done, episode_step = False, 0

            while not done:
                episode_step += 1
                (mu, sigma), value = worker_model(torch.from_numpy(state))
                prob = worker_model.distribution(mu, sigma)
                action = prob.sample().detach()
                log_prob = prob.log_prob(action)
                entropy = prob.entropy().mean()
                if rank == 0: # TODO: delete
                    print('mu: {}, sigma: {}, action: {}'.format(mu.data, sigma.data, action.data))

                state, reward, terminal, truncated, _ = self.env.step(
                    action.numpy().clip(-self.max_action, self.max_action))
                done = terminal or truncated

                with self.lock:
                    self.total_step.value += 1

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    break
            
            R = torch.zeros(1, 1)
            if not done:
                _, value = worker_model(torch.from_numpy(state))
                R = value.detach()
            values.append(R)

            policy_loss, value_loss = 0, 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)

                delta_t = rewards[i] + self.gamma * \
                    values[i + 1] - values[i]
                gae = gae * self.gamma * self.lambd + delta_t
                policy_loss -= (log_probs[i] * gae.detach() + self.beta * entropies[i])

            self.optimizer.zero_grad()
            (policy_loss + self.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                worker_model.parameters(), self.max_grad_norm)
            for global_param, worker_param in zip(self.global_model.parameters(), worker_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = worker_param.grad
            self.optimizer.step()

            reward_list.append(np.sum(rewards))
            if rank == 0:
                print('Episode: {}, Reward: {}'.format(
                    len(reward_list), reward_list[-1]))
            
        plt.figure()
        plt.plot(reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.config['images_path'] + '/reward_train_' + str(rank) + '.png')

        #         if t % self.t_max == 0 or done:
        #             _, value = worker_model(torch.from_numpy(next_state))
        #             R_buf = []
        #             R = 0 if terminal else value.data.numpy()[0]
        #             for r in reward_buf[::-1]:
        #                 R = r + self.gamma * R
        #                 R_buf.append(R)
        #             R_buf.reverse()

        #             loss = worker_model.loss(torch.from_numpy(np.vstack(state_buf)),
        #                                     torch.from_numpy(np.vstack(action_buf)),
        #                                     torch.from_numpy(np.array(R_buf)),
        #                                     self.beta)

        #             self.optimizer.zero_grad()
        #             loss.backward()
        #             torch.nn.utils.clip_grad_norm_(worker_model.parameters(), self.max_grad_norm)
        #             for global_param, worker_param in zip(self.global_model.parameters(), worker_model.parameters()):
        #                 if global_param.grad is None:
        #                     global_param._grad = worker_param.grad
        #             self.optimizer.step()
        #             worker_model.load_state_dict(self.global_model.state_dict())



        #             state_buf, action_buf, reward_buf = [], [], []

        #             if done:
        #                 r_list.append(total_reward) 
        #                 with self.lock:
        #                     self.r_list.append(total_reward)
        #                 with self.total_step.get_lock():
        #                     self.total_step.value += 1
                        
        #                 if rank == 0:
        #                     print('Total steps: {}/{}, Reward: {:.3f}'.format(
        #                         self.total_step.value, self.T_max, total_reward))
        #                 break

        #         state = next_state
        #         t += 1

        # plt.figure()
        # plt.plot(r_list)
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.savefig(self.config['images_path'] +
        #             '/reward_train_' + str(rank) + '.png')
