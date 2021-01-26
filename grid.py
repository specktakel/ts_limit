import numpy as np


class Grid():
    def __init__(self):
        g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
        m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
        grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
        for i in range(g_space.shape[0]):
            for j in range(m_space.shape[0]):
                grid[i, j, :] = g_space[i], m_space[j]
        grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
        self.grid = grid



    def get_gm(self, arg, shape=None):
        '''arg should be a tuple or single int.'''
        if shape is not None:
            grid = self.grid.reshape(shape)
        else:
            grid = self.grid
        return grid[arg]


    def get_grid(self, shape=None):
        if shape is not None:
            grid = self.grid.reshape(shape)
        else:
            grid = self.grid
        return grid

