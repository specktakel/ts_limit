import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rc
from scipy.stats import ncx2
from scipy.optimize import curve_fit

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
'''Script to do binning and analysis of highest (?) TS value per pseudo experiment (PE).
'''
data_H0 = -421745.9637820099
def fit_func(x, df, nc):
    return ncx2.pdf(x, df, nc)

# Maybe do "running" analysis as data comes in
# Check for folders with full ts_95 coverage
cwd = sys.path[0]
print(cwd)
'''
# read in loglike of null hypothesis H0_{i}
H0_list = os.listdir(f'{cwd}/../roi_simulation/loglikes')
# print(H0_list)
H0 = np.zeros((len(H0_list)))
for c, v in enumerate(H0_list):
    H0[c] = np.loadtxt(v)

# np.savetxt('~/H0_loglikes.dat', H0)
'''
H0 = np.loadtxt('H0_loglikes.dat')
print(H0)
fig_1 = plt.figure(1, dpi=150)    # loglike H0 figure
ax_1 = fig_1.add_subplot(111)
ax_1.set_title('Loglikes of simulated ROI, nbins=20, n=100')
n_1, bins_1, patches_1 = ax_1.hist(H0, bins=20, label='Simulated ROI', density=True)
ax_1.plot((data_H0, data_H0), (0, .00040), label='real data')
ax_1.set_xlabel('loglike of nullhypothesis')
ax_1.set_ylabel('probability density')
ax_1.legend()

fig_2 = plt.figure(2, dpi=150)
ax_2 = fig_2.add_subplot(111)
ax_2.set_title('Distribution of largest TS per PE')
ts_max_values = np.loadtxt('ts_max_values.dat')
n_2, bins_2, patches_2 = ax_2.hist(ts_max_values, bins=20, label='max ts values per PE')
ax_2.legend()
delta_x = bins_2[1] - bins_2[0]
x = np.linspace(bins_2[0] + delta_x / 2, bins_2[-1] - delta_x / 2, num = len(n_2), endpoint=True)
n_2[0] = 0
print(x, n_2)
popt, pcov = curve_fit(fit_func, x, n_2)

