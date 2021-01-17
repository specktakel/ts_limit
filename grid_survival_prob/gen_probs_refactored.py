from probs import Probs

import numpy as np
import os
import sys
import psutil
process = psutil.Process(os.getpid())
cwd = os.getcwd()


which_gm = int(sys.argv[1])
which_roi = int(sys.argv[2])
nsim = int(sys.argv[3])

seed = None # is preset
B0 = 10.    # is preset
ppb = 10    # is preset


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
#outpath = f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/new_probs/roi_{which_roi}"
outpath = cwd

save_path = f'{cwd}/prob_memory_test.dat'
'''Create out directory if necessary.'''
try:
    print(f"making directory {outpath}")
    os.mkdir(outpath)
except FileExistsError:
    print(f"{outpath} already exists")


probs = Probs(log10MeV, g, m)
p = np.zeros((nsim, log10MeV.shape[0]))
for i in range(nsim):
    p[i] = probs.propagation()
    print("memory usage:", process.memory_info().rss * 1e-6)
    probs.reload_module()
    print(i, "done")

try:
    data = np.loadtxt(save_path)
    outdata = np.append(data, p, axis=0)
except:
    outdata = p

np.savetxt(save_path, p)




