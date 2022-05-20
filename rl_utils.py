# Author: Mattia Silvestri

"""
    Utility script with methods and classes for the RL algorithms and environments.
"""

import gym
import numpy as np
from utility import PLSInstance
from latinsq import LatinSquare

########################################################################################################################


class PLSEnv(gym.Env):
    """
    Gym wrapper for the PLS.
    Attributes:
        dim: int; PLS size.
        instance: PLSInstance; the PLS instance.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self, dim):
        super(PLSEnv, self).__init__()

        self._dim = dim
        self.action_space = gym.spaces.Discrete(dim**3)
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.int8, shape=(dim**3, ))
        self._instance = PLSInstance(n=dim)

    @property
    def dim(self):
        """
        The PLS size.
        :return: int; the PLS size.
        """
        return self._dim

    def step(self, action):
        """
        A step in the environment.
        :param action: int; integer corresponding to the PLS cell and value to assign.
        :return: numpy.array, float, boolean, dict; observations, reward, end of episode flag and additional info.
        """
        assert 0 <= action < self.action_space.n, "Out of actions space"
        x_coor, y_coor, val = np.unravel_index(action, shape=(self._dim, self._dim, self._dim))
        feasible = self._instance.assign(cell_x=x_coor, cell_y=y_coor, num=val)
        count_assigned_vars = np.sum(self._instance.square)
        solved = False
        if feasible:
            reward = -1
            if count_assigned_vars == self._dim ** 2:
                done = True
                solved = True
            else:
                done = False
        else:
            reward = -1000
            done = True

        obs = self._instance.square.reshape(-1)
        info = dict()
        info['Feasible'] = feasible
        info['Num. assigned vars'] = count_assigned_vars
        info['Solved'] = solved

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment.
        :return: numpy.array; the observations.
        """
        self._instance = PLSInstance(n=self._dim)
        #obs = self._instance.square.reshape(-1, 1)
        obs = self._instance.square.reshape(-1)
        
        return obs

    def render(self, mode="human"):
        """
        Visualize the PLS assignments.
        :param mode:
        :return:
        """
        self._instance.visualize()


class PLSEnv_partially_filled(gym.Env):
    """
    Gym wrapper for the PLS.
    Attributes:
        dim: int; PLS size.
        instance: PLSInstance; the PLS instance.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }
    

    def __init__(self, dim, perc_to_fill=None):
        super(PLSEnv_partially_filled, self).__init__()

        self._dim = dim
        self.action_space = gym.spaces.Discrete(dim**3)
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.int8, shape=(dim**3, ))
        self._perc_to_fill_list = [0.25, 0.50, 0.75]
        
        if perc_to_fill!=None and (perc_to_fill<0 or perc_to_fill>1):
            raise Exception("perc_to_fill must be a value between 0 and 1.")
        self._perc_to_fill = perc_to_fill

    @property
    def dim(self):
        """
        The PLS size.
        :return: int; the PLS size.
        """
        return self._dim

    def step(self, action):
        """
        A step in the environment.
        :param action: int; integer corresponding to the PLS cell and value to assign.
        :return: numpy.array, float, boolean, dict; observations, reward, end of episode flag and additional info.
        """
        assert 0 <= action < self.action_space.n, "Out of actions space"
        x_coor, y_coor, val = np.unravel_index(action, shape=(self._dim, self._dim, self._dim))
        feasible = self._instance.assign(cell_x=x_coor, cell_y=y_coor, num=val)
        count_assigned_vars = np.sum(self._instance.square)
        solved = False
        if feasible:
            reward = 0
            if count_assigned_vars == self._dim ** 2:
                done = True
                solved = True
                reward = 1
            else:
                done = False
        else:
            reward = -1
            done = True

        obs = self._instance.square.reshape(-1)
        info = dict()
        info['Feasible'] = feasible
        info['Num. assigned vars'] = count_assigned_vars
        info['Solved'] = solved

        return obs, reward, done, info
    
    def reset(self):
        """
        Reset the environment and construct a new partially filled latin square.
        :return: numpy.array; the observations.
        """
        self._instance = PLSInstance(n=self._dim)
        
        total_nums = self._dim ** 2
        if self._perc_to_fill==None:
            #Sample a percentage at random. (used during training)
            p = np.random.choice(self._perc_to_fill_list)
        else:
            p = self._perc_to_fill
        # Sample the positions of the latin square to fill
        total_grid_cells = self._dim ** 2
        n_cells_to_fill = int(p*self._dim ** 2)
        cells_to_fill = np.random.choice(total_grid_cells, n_cells_to_fill, replace=False)
        # Generate a full latin square
        ls = LatinSquare.random(n=self._dim)
        
        for cell in cells_to_fill:
            x_coor, y_coor = np.unravel_index(cell, shape=(self._dim, self._dim))
            value_to_insert = ls[x_coor, y_coor] - 1
            feasible = self._instance.assign(cell_x=x_coor, cell_y=y_coor, num=value_to_insert)
            assert feasible, "Error during latin square initialization"
        
        obs = self._instance.square.reshape(-1)
        
        return obs

    def render(self, mode="human"):
        """
        Visualize the PLS assignments.
        :param mode:
        :return:
        """
        self._instance.visualize()

########################################################################################################################


class RandomAgent:
    """
    Agent that randomly choose an actions according to the actions space.
    """
    def __init__(self, num_actions, render_actions=False):
        self._num_actions = num_actions
        self._dim = self._check_is_perfect_cube()
        self._render_actions = render_actions

    def _check_is_perfect_cube(self):
        """
        Check that the number of actions is a perfect cube, namely dim^3.
        :return: int; the size of the PLS.
        """
        c = int(self._num_actions ** (1 / 3.))

        if c ** 3 == self._num_actions:
            return c
        elif (c + 1) ** 3 == self._num_actions:
            return c + 1
        else:
            raise Exception("Actions space must be a perferct cube")

    @property
    def num_actions(self):
        """
        Actions space.
        :return: int; the number of actions in the actions space.
        """
        return self._num_actions

    def act(self):
        """
        Choose an action.
        :return: int; the chosen action.
        """
        action = np.random.randint(low=0, high=self._num_actions, size=1)
        if self._render_actions:
            row, col, val = np.unravel_index(action, shape=(self._dim, self._dim, self._dim))
            print(f'Row: {row+1} | Column: {col+1} | Value: {val+1}')

        return action


########################################################################################################################

if __name__ == '__main__':
    env = PLSEnv(dim=2)
    agent = RandomAgent(num_actions=env.action_space.n, render_actions=True)

    done = False

    while not done:
        action = agent.act()
        obs, reward, done, info = env.step(action=action)
        env.render()
        if info['Solved']:
            print("Solved!")
        print('\n' + '-'*100 + '\n')