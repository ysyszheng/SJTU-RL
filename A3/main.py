import random


class Grid:
    def __init__(self, position, value=.0, is_start=False, is_goal=False, is_cliff=False):
        self.val = value
        self.pos = position
        self.act = None
        self.is_start = is_start
        self.is_goal = is_goal
        self.is_cliff = is_cliff


class Cliff:
    def __init__(self, width, height, start, goal, cliff_list, gamma=1, r=-1, r_cliff=-100):
        self.w = width
        self.h = height
        self.start = start
        self.goal = goal
        self.cliff_list = cliff_list
        self.gamma = gamma
        self.r = r
        self.r_cliff = r_cliff
        self.grid_list = []
        for i in range(width*height):
            self.grid_list.append(
                Grid(i, is_start=i == start, is_goal=i == goal, is_cliff=i in cliff_list))

    def __str__(self) -> str:
        grid_str = ''
        for i in range(self.w*self.h):
            if i % self.w == 0:
                grid_str += '| '
            if self.grid_list[i].is_start:
                grid_str += 'S'
            elif self.grid_list[i].is_goal:
                grid_str += 'G'
            elif self.grid_list[i].is_cliff:
                grid_str += 'x'
            elif self.grid_list[i].act == None:
                grid_str += 'O'
            else:
                grid_str += self.grid_list[i].act
            if (i+1) % self.w == 0:
                grid_str += ' |\n'
            else:
                grid_str += ' | '
        return grid_str

    def epsilon_greedy(self, Q, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return Q.index(max(Q))

    def step(self, s, a):
        if a == 0:
            s_next = s - self.w if s >= self.w else s
        elif a == 1:
            s_next = s + 1 if (s+1) % self.w != 0 else s
        elif a == 2:
            s_next = s + self.w if s < self.w*(self.h-1) else s
        elif a == 3:
            s_next = s - 1 if s % self.w != 0 else s
        if s_next in self.cliff_list:
            return self.start, self.r_cliff
        return s_next, self.r

    def sarsa(self, epsilon=.1, alpha=.2, num_episodes=10000):
        Q = [[0 for _ in range(4)] for _ in range(self.w*self.h)]
        for _ in range(num_episodes):
            s = self.start
            a = self.epsilon_greedy(Q[s], epsilon)
            while not self.grid_list[s].is_goal:
                s_next, r = self.step(s, a)
                a_next = self.epsilon_greedy(Q[s_next], epsilon)
                Q[s][a] += alpha * \
                    (r + self.gamma * Q[s_next][a_next] - Q[s][a])
                s = s_next
                a = a_next
        s = self.start
        while s != self.goal:
            self.grid_list[s].act = ['^', '>', 'v', '<'][Q[s].index(max(Q[s]))]
            s, _ = self.step(s, Q[s].index(max(Q[s])))

    def q_learning(self, epsilon=.1, alpha=.2, num_episodes=10000):
        Q = [[0 for _ in range(4)] for _ in range(self.w*self.h)]
        for _ in range(num_episodes):
            s = self.start
            while not self.grid_list[s].is_goal:
                a = self.epsilon_greedy(Q[s], epsilon)
                s_next, r = self.step(s, a)
                Q[s][a] += alpha * \
                    (r + self.gamma * max(Q[s_next]) - Q[s][a])
                s = s_next
        s = self.start
        while s != self.goal:
            self.grid_list[s].act = ['^', '>', 'v', '<'][Q[s].index(max(Q[s]))]
            s, _ = self.step(s, Q[s].index(max(Q[s])))


if __name__ == '__main__':
    eps = 1
    Cliff1 = Cliff(12, 4, start=36, goal=47, cliff_list=range(37, 47))
    print('\n Cliff Walking Env:')
    print(Cliff1)
    print(f'\n Sarsa: epsilon={eps}')
    Cliff1.sarsa(epsilon=eps, num_episodes=100000)
    print(Cliff1)
    Cliff2 = Cliff(12, 4, start=36, goal=47, cliff_list=range(37, 47))
    print(f'\n Q-Learning: epsilon={eps}')
    Cliff2.q_learning(epsilon=eps)
    print(Cliff2)
