import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

'''Script to do binning and analysis of highest (?) TS value per pseudo experiment (PE).
'''


# Maybe do "running" analysis as data comes in
# Check for folders with full ts_95 coverage
cwd = sys.path[0]
print(cwd)

# read in loglike of null hypothesis H0_{i}
H0_list = os.listdir(f'{cwd}/../roi_simulation/loglikes')
# print(H0_list)
H0 = np.zeros((len(H0_list)))
for c, v in enumerate(H0_list):
    H0[c] = np.loadtxt(f'{cwd}/../roi_simulation/loglikes/{v}')

print(H0)
np.savetxt('H0_loglikes.dat', H0)

fig_1 = plt.figure(1)    # loglike H0 figure
ax_1 = fig_1.add_subplot(111)

n_1, bins_1, patches_1 = ax_1.hist(H0)

