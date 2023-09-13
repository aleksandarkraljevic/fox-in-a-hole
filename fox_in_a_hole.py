import numpy as np

class FoxInAHole():
    def __init__(self, n_holes):
        self.n_holes = n_holes

    def reset(self):
        # reset the environment to initial state
        self.reward = 0
        self.done = [False, False] # first one stands for won, second one for lost
        self.observation = [-1]*(2*self.n_holes)
        self.fox = np.random.randint(1, self.n_holes+1)
        return self.observation, self.fox

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
        if timestep <= len(self.observation):
            self.observation[timestep-1] = action
            if timestep == len(self.observation):
                self.done = [False, True] # the game is lost if the fox hasn't been found after 2*n_holes timesteps
            if action == self.fox:
                self.done = [True, False] # the game is won when the fox is found
            else:
                self.reward -= 1
        else:
            print('Timestep is too large for the memory size.')
        return self.observation, self.reward, self.done