class Grid:
    def __init__(self, position, value=.0, is_terminal=False):
        self.val = value
        self.pos = position
        self.act = {'n': .0, 'e': .0, 's': .0, 'w': .0} if is_terminal else {
            'n': 0.25, 'e': 0.25, 's': 0.25, 'w': 0.25}
        self.is_terminal = is_terminal


class GridWorld:
    def __init__(self, width, height, terminal_list, gamma, theta=.0001):
        self.w = width
        self.h = height
        self.terminal_list = terminal_list
        self.gamma = gamma
        self.theta = theta
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

    def act_val(self, pos, action):
        grid = self.grid_list[pos]
        val = {
            "n": self.r + self.gamma *
            (self.grid_list[pos-self.w].val if pos >=
             self.w else grid.val),
            "e": self.r + self.gamma *
            (self.grid_list[pos+1].val if (pos+1) %
             self.w else grid.val),
            "s": self.r + self.gamma *
            (self.grid_list[pos+self.w].val if (pos +
                                                self.w) < self.w*self.h else grid.val),
            "w": self.r + self.gamma *
            (self.grid_list[pos-1].val if pos % self.w else grid.val)
        }
        return val.get(action, lambda: "Invalid action")

    def eval(self):
        while True:
            delta = .0
            update_val = []
            for i in range(self.w*self.h):
                grid = self.grid_list[i]
                if grid.is_terminal:
                    update_val.append(.0)
                else:
                    v = grid.val
                    val = grid.act['n']*self.act_val(i, 'n')+grid.act['e'] * \
                        self.act_val(i,
                                     'e')+grid.act['s']*self.act_val(i, 's')+grid.act['w']*self.act_val(i, 'w')
                    delta = max(delta, abs(val-v))
                    update_val.append(val)
            for i in range(self.w*self.h):
                self.grid_list[i].val = update_val[i]
            if delta < self.theta:
                for i in range(self.w*self.h):
                    grid = self.grid_list[i]
                    if grid.is_terminal:
                        continue
                    for action in ['n', 'e', 's', 'w']:
                        grid.act[action] = .0
                    else:
                        act_val_lst = [self.act_val(i, 'n'), self.act_val(
                            i, 'e'), self.act_val(i, 's'), self.act_val(i, 'w')]
                        value = max(act_val_lst)
                        act = [idx for idx, val in enumerate(
                            act_val_lst) if val == value]
                        pr = 1/len(act)
                        for action in [list(grid.act)[j] for j in act]:
                            grid.act[action] = pr
                break

    def policy_iter(self):
        while True:
            while True:
                delta = 0
                update_val = []
                for i in range(self.w*self.h):
                    grid = self.grid_list[i]
                    if grid.is_terminal:
                        update_val.append(.0)
                    else:
                        v = grid.val
                        val = grid.act['n']*self.act_val(i, 'n')+grid.act['e'] * \
                            self.act_val(i,
                                         'e')+grid.act['s']*self.act_val(i, 's')+grid.act['w']*self.act_val(i, 'w')
                        delta = max(delta, abs(val-v))
                        update_val.append(val)
                for i in range(self.w*self.h):
                    self.grid_list[i].val = update_val[i]
                if delta < self.theta:
                    break
            is_stable = True
            for i in range(self.w*self.h):
                grid = self.grid_list[i]
                if grid.is_terminal:
                    continue
                old_action = grid.act.copy()
                for action in ['n', 'e', 's', 'w']:
                    grid.act[action] = .0
                act_val_lst = [self.act_val(i, 'n'), self.act_val(
                    i, 'e'), self.act_val(i, 's'), self.act_val(i, 'w')]
                act = [idx for idx, val in enumerate(
                    act_val_lst) if val == max(act_val_lst)]
                pr = 1/len(act)
                for action in [list(grid.act)[j] for j in act]:
                    grid.act[action] = pr
                if grid.act != old_action:
                    is_stable = False
            if is_stable:
                break

    def value_iter(self):
        while True:
            delta = 0
            update_val = []
            for i in range(self.w*self.h):
                grid = self.grid_list[i]
                if grid.is_terminal:
                    update_val.append(.0)
                else:
                    v = grid.val
                    act_val_lst = [self.act_val(i, 'n'), self.act_val(
                        i, 'e'), self.act_val(i, 's'), self.act_val(i, 'w')]
                    max_val = max(act_val_lst)
                    update_val.append(max_val)
                    delta = max(delta, abs(v-max_val))
            for i in range(self.w*self.h):
                self.grid_list[i].val = update_val[i]
            if delta < self.theta:
                break
        for i in range(self.w*self.h):
            grid = self.grid_list[i]
            if grid.is_terminal:
                continue
            for action in ['n', 'e', 's', 'w']:
                grid.act[action] = .0
            act_val_lst = [self.act_val(i, 'n'), self.act_val(
                i, 'e'), self.act_val(i, 's'), self.act_val(i, 'w')]
            act = [idx for idx, val in enumerate(
                act_val_lst) if val == max(act_val_lst)]
            pr = 1/len(act)
            for action in [list(grid.act)[j] for j in act]:
                grid.act[action] = pr


if __name__ == '__main__':
    World1 = GridWorld(6, 6, [1,35], 1.0)
    World1.eval()
    print('Iterative Policy Evaluation')
    World1.print_val()
    World1.print_policy()
    World2 = GridWorld(6, 6, [1, 35], 1.0)
    World2.policy_iter()
    print('\nPolicy Iteration')
    World2.print_val()
    World2.print_policy()
    World3 = GridWorld(6, 6, [1, 35], 1.0)
    World3.value_iter()
    print('\nValue Iteration')
    World3.print_val()
    World3.print_policy()
