from ts_limit.grid_survival_prob.probs import Probs

import numpy as np
import os
import sys
import psutil
process = psutil.Process(os.getpid())
cwd = os.getcwd()


# which_gm = int(sys.argv[1])
# which_roi = int(sys.argv[2])
# nsim = int(sys.argv[3])
which_gm = 645
seed = 1  # is preset
B0 = 10.    # is preset
# ppb = 10    # is preset

oversampling = [i for i in range(7, 16)]
'''Set up parameter list'''
g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
for i in range(g_space.shape[0]):
    for j in range(m_space.shape[0]):
        grid[i, j, :] = g_space[i], m_space[j]
grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
g = grid[which_gm, 0]
m = grid[which_gm, 1]


'''Energy binning related stuff goes here'''
log10MeV = np.loadtxt(cwd+'/energy_bins.dat')


'''Create out directory if necessary.'''
for i in oversampling:
    probs = Probs(log10MeV, g, m, seed=1, ppb=i)
    p = probs.fractured_propagation(seed=1)
    np.savetxt(f'ppb_testing/ppb_{i}.dat', p)
