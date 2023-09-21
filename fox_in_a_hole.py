import numpy as np

class FoxInAHole():
    def __init__(self, n_holes, memory_size):
        self.n_holes = n_holes
        self.memory_size = memory_size

    def reset(self):
        # reset the environment to initial state
        self.reward = 0
        self.done = [False, False] # first one stands for won, second one for lost
        self.fox = np.random.randint(1, self.n_holes+1)
        return self.fox, self.done

    def step(self):
        # perform one step in the game logic
        if self.fox == 1:
            self.fox += 1
        elif self.fox == self.n_holes:
            self.fox -= 1
        else:
            random_movement = np.random.random()
            if random_movement < 0.5:
                self.fox -= 1
            else:
                self.fox += 1
        return self.fox

    def guess(self, action, timestep):
        # perform one guess in the game logic
        if action == self.fox:
            self.done = [True, False] # the game is won when the fox is found
            self.reward += 2
        elif timestep == self.memory_size:
            self.done = [False, True] # the game is lost if the fox hasn't been found after the max amount of timesteps
        self.reward -= 1
        return self.reward, self.done