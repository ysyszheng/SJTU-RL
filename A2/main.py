import random


class Grid:
    def __init__(self, position, value=.0, is_terminal=False):
        self.val = value
        self.pos = position
        self.act = {'n': .0, 'e': .0, 's': .0, 'w': .0} if is_terminal else {
            'n': 0.25, 'e': 0.25, 's': 0.25, 'w': 0.25}
        self.is_terminal = is_terminal


class GridWorld:
    def __init__(self, width, height, terminal_list, gamma):
        self.w = width
        self.h = height
        self.terminal_list = terminal_list
        self.gamma = gamma
        self.r = -1
        self.grid_list = []
        for i in range(width*height):
            self.grid_list.append(Grid(i, is_terminal=i in terminal_list))

    def print_val(self):
        for i in range(self.w*self.h):
            if i % self.w == 0 and i != 0:
                print(f'\n{self.grid_list[i].val:.2f}\t', end='')
            elif i != self.w*self.h-1:
                print(f'{self.grid_list[i].val:.2f}\t', end='')
            else:
                print(f'{self.grid_list[i].val:.2f}\t')

    def print_policy(self):
        arrows = ['↑', '→', '↓', '←']
        for i in range(self.w*self.h):
            mov = ''
            for j in range(4):
                if list(self.grid_list[i].act.items())[j][1] != .0:
                    mov += arrows[j]
            if i % self.w == 0 and i != 0:
                print(f'\n{mov}\t', end='')
            elif i != self.w*self.h-1:
                print(f'{mov}\t', end='')
            else:
                print(f'{mov}\t')

    def generate_episode(self):
        episode = []
        start = random.randint(0, self.w*self.h-1)
        while self.grid_list[start].is_terminal:
            start = random.randint(0, self.w*self.h-1)
        episode.append(start)
        while True:
            if self.grid_list[start].is_terminal:
                break
            mov = random.choices(
                list(self.grid_list[start].act.keys()), list(self.grid_list[start].act.values()))[0]
            if mov == 'n':
                start -= self.w if start >= self.w else 0
            elif mov == 'e':
                start += 1 if start % self.w != self.w-1 else 0
            elif mov == 's':
                start += self.w if start < self.w*(self.h-1) else 0
            elif mov == 'w':
                start -= 1 if start % self.w != 0 else 0
            episode.append(start)
        return episode

    def first_visit_MC(self, episode_num=10000, alpha=.01):
        while episode_num:
            episode_num -= 1
            episode = self.generate_episode()
            G = 0
            for i in range(len(episode)-2, -1, -1):
                G = self.gamma*G+self.r
                if episode[i] not in episode[:i]:
                    self.grid_list[episode[i]].val += alpha * \
                        (G-self.grid_list[episode[i]].val)

    def every_visit_MC(self, episode_num=10000, alpha=.01):
        while episode_num:
            episode_num -= 1
            episode = self.generate_episode()
            G = 0
            for i in range(len(episode)-2, -1, -1):
                G = self.gamma*G+self.r
                self.grid_list[episode[i]].val += alpha * \
                    (G-self.grid_list[episode[i]].val)

    def TD(self, episode_num=10000, alpha=.01):
        while episode_num:
            episode_num -= 1
            episode = self.generate_episode()
            for i in range(len(episode)-1):
                self.grid_list[episode[i]].val += alpha * \
                    (self.r+self.gamma*self.grid_list[episode[i+1]].val -
                        self.grid_list[episode[i]].val)


if __name__ == '__main__':
    gw1 = GridWorld(6, 6, [1, 35], 1)
    print('\nFirst-Visit Monte-Carlo Policy Evaluation')
    gw1.first_visit_MC()
    gw1.print_val()
    gw2 = GridWorld(6, 6, [1, 35], 1)
    print('\nEvery-Visit Monte-Carlo Policy Evaluation')
    gw2.every_visit_MC()
    gw2.print_val()
    gw3 = GridWorld(6, 6, [1, 35], 1)
    print('\nTemporal-Difference Policy Evaluation')
    gw3.TD()
    gw3.print_val()
