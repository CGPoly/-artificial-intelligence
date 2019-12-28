import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self):
        self.labyrinth = np.asarray([[None, None, None, None], [None, 0, None, -1], [None, None, None, 1]], np.ndarray)
        self.values = np.zeros((self.labyrinth.shape[0], self.labyrinth.shape[1], 4))
        self.acc_pos = [0, 0]
        self.live_penalty = -0.1
        self.learning_rate = 0.01
        self.probability = 0.8
        self.discount = 0.9
        self.step = 0
        self.update = 1000
        self.epoch = 0
        self.fig, self.ax = plt.subplots(2)
        self.text = []
        while self.epoch < 50000:
            self.move_agent()
        self.translate_values()
    
    def deterministic_move(self, direction: int) -> float:
        direction %= 4
        if direction < 2:
            self.acc_pos[direction % 2] += 1
        else:
            self.acc_pos[direction % 2] -= 1
        if not (-1 < self.acc_pos[0] < self.labyrinth.shape[0] and -1 < self.acc_pos[1] < self.labyrinth.shape[1]):
            self.deterministic_move(direction + 2)
        if self.labyrinth[self.acc_pos[0], self.acc_pos[1]] == 0:
            self.deterministic_move(direction + 2)
        if self.labyrinth[self.acc_pos[0], self.acc_pos[1]] is not None:
            tmp = self.labyrinth[self.acc_pos[0], self.acc_pos[1]]
            self.acc_pos = [0, 0]
            self.epoch += 1
            return tmp
        return self.live_penalty
    
    def move(self, direction: int) -> float:
        self.step += 1
        if self.step % self.update == 0:
            self.translate_values()
        rand = np.random.rand()
        if rand < (1 - self.probability) / 2:
            direction -= 1
        elif 1 - ((1 - self.probability) / 2) < rand:
            direction += 1
        return self.deterministic_move(direction)
    
    def give_value_after_move(self, direction: int) -> list:
        tmp_pos = self.acc_pos.copy()
        direction %= 4
        if direction < 2:
            tmp_pos[direction % 2] += 1
        else:
            tmp_pos[direction % 2] -= 1
        if not (-1 < tmp_pos[0] < self.labyrinth.shape[0] and -1 < tmp_pos[1] < self.labyrinth.shape[1]):
            return self.values[self.acc_pos[0], self.acc_pos[1]]
        if self.labyrinth[tmp_pos[0], tmp_pos[1]] == 0:
            return self.values[self.acc_pos[0], self.acc_pos[1]]
        if self.labyrinth[tmp_pos[0], tmp_pos[1]] is not None:
            return [self.labyrinth[tmp_pos[0], tmp_pos[1]]]
        return self.values[tmp_pos[0], tmp_pos[1]]
    
    def move_agent(self):
        move_dir: int
        acc_values = self.values[self.acc_pos[0], self.acc_pos[1]].copy()
        if any([[i != j and acc_values[i] == acc_values[j] for j in range(4)] for i in range(4)]):
            possible = np.where(acc_values == max(acc_values))[0]
            move_dir = possible[np.random.randint(0, len(possible))]
        else:
            move_dir = acc_values.index(max(acc_values))
        tmp_pos = self.acc_pos.copy()
        reward = self.move(int(move_dir))
        # evaluation before the action
        # self.values[tmp_pos[0], tmp_pos[1], move_dir] = (1-self.learning_rate) * self.values[tmp_pos[0], tmp_pos[1], move_dir] \
        #     + self.learning_rate * (reward + self.discount * sum([((1 - self.probability) / 2) * max(self.give_value_after_move(move_dir-1)),
        #                                                           self.probability*max(self.give_value_after_move(move_dir)),
        #                                                           ((1 - self.probability) / 2)*max(self.give_value_after_move(move_dir+1))]))
        # evaluation after the action (standard formula)
        self.values[tmp_pos[0], tmp_pos[1], move_dir] = (1 - self.learning_rate) * self.values[tmp_pos[0], tmp_pos[1], move_dir] \
            + self.learning_rate * (reward + self.discount * max(self.values[self.acc_pos[0], self.acc_pos[1]]))
        
    def translate_values(self):
        self.ax[0].clear()
        self.ax[1].clear()
        image = [[[0, 0, 255] if x == self.acc_pos[0] and y == self.acc_pos[1] else (
            [255, 255, 255] if self.labyrinth[x, y] is None
            else ([50, 50, 50] if self.labyrinth[x, y] == 0
                  else ([255, 0, 0] if self.labyrinth[x, y] == -1 else [0, 255, 0]))) for y in range(self.labyrinth.shape[1])] for x in range(self.labyrinth.shape[0])]
        self.ax[0].imshow(image, origin="bottom")
        self.ax[1].imshow(image, origin="bottom")
        for i in range(self.labyrinth.shape[0]):
            for j in range(self.labyrinth.shape[1]):
                if self.labyrinth[i, j] is None:
                    tmp_0 = "\\" + str(round(self.values[i, j, 0], 3 if self.values[i, j, 0] >= 0 else 2)) + "/" + "\n" + \
                        "\\   /\n" + \
                        str(round(self.values[i, j, 3], 3 if self.values[i, j, 0] >= 0 else 2)) + " X " + str(round(self.values[i, j, 1], 3 if self.values[i, j, 0] >= 0 else 2)) \
                        + "\n/   \\\n" + \
                        "/" + str(round(self.values[i, j, 2], 3 if self.values[i, j, 0] >= 0 else 2)) + "\\"
                    index = list(self.values[i, j, :]).index(max(self.values[i, j]))
                    tmp_1 = "/\\\n/  \\\n/    \\" if index == 0 else ("\\\n  \\\n  /\n/" if index == 1 else ("\\    /\n\\  /\n\\/" if index == 2 else "  /\n/\n\\\n  \\"))
                    self.ax[0].text(j, i, tmp_0, ha="center", va="center", color='black')
                    self.ax[1].text(j, i, tmp_1, ha="center", va="center", color='black')
        plt.pause(1e-16)


if __name__ == "__main__":
    Environment()
    print("finished")
    plt.show()
