#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:53:44 2020

@author: julian
"""
# import packages
import numpy as np
import sys
# sys.path.append('../outdata')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from os import listdir
from os.path import isfile
from matplotlib import rc
import re

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
which_roi = int(sys.argv[1])

directory = f'roi_{which_roi}/ts_95'
prefix = 'ts_95_'
postfix = '.dat'

file_list = listdir(directory)
#print(file_list)
data_list = []
for c, v in enumerate(file_list):
    if prefix in v and postfix in v:
        data_list.append(f'{directory}/{v}')
##  print(data_list)





likes = np.zeros((900))    # indices: files, simulation, value
# for file in data_list:
#     with open(file, mode='r') as f:
#         next(f)
#         for c, v in enumerate(f):
#             # print(v)
#             v= v.rstrip('\n')
#             v = v.split(' ')
#             # print(v)
#             num = re.sub('\.dat$', '', file)
#             num = int(re.sub('^'+directory+'bird', '', num))
#             likes[num, c, 0] = np.float(v[0])    # log L_1
#             likes[num, c, 1] = np.float(v[-1])   # log L_0
for file in data_list:
    name = file.rsplit('_', maxsplit=1)
    num = int(name[1].rstrip('.dat'))
    # print(num)
    temp = np.loadtxt(file)
    # ts = 2 * np.sort(temp[:, 0] - temp[:, -1])
    ts_95 = temp
    likes[num] = ts_95

likes = likes.reshape((30, 30))


# for file in sm_data_list:
#     print(file)
#     num = file.lstrip('bird')
#     print(num)
#     with open(sm_dir+file, mode='r') as f:
#         next(f)
#         for c, v in enumerate(f):
#             # print(v)
#             v= v.rstrip('\n')
#             v = v.split(' ')
#             #print(v[0])

#             likes[num, c, 0] = np.float(v[0])
#             likes[num, c, 1] = np.float(v[-1])
            # print(num)
#ratio = likes[:, :, 0] / likes[:, :, 1]
#print(ratio)
# get test statistics: TS = -2(init_loglike - final_loglike)
#ts = np.zeros((900, 100))
#ts = - 2 * np.log(ratio)
#ts = 2 * (likes[:, :, 1] - likes[:, :, 0])
#ts = np.sort(ts)
# print(ts)
#ts_test = 2 * (likes_test[:, :, 0] - likes_test[:, :, 1])
#ts_test = np.sort(ts_test)
#ts_plot_test = np.zeros((900))
#ts_plot_test = ts_test[:, 94]
#ts_plot_test = ts_plot_test.reshape((30, 30))
# get plottable TS values, here 95% value (so the 95th entry?! ask Dieter)
#ts_plot = np.zeros((900))
#ts_plot = ts[:, 94]
#ts_95 = ts_plot.copy()
#ts_plot = ts_plot.reshape((30, 30))
#print(ts_plot)

plot_this = np.zeros((900))


# for reference, the lines from sim.py:
'''
g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
for i in range(g_space.shape[0]):
    for j in range(m_space.shape[0]):
        grid[i, j, :] = g_space[i], m_space[j]
grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
xv, yv = np.meshgrid(m_space, g_space)
#print(xv)
xv = xv.reshape((900))
yv = yv.reshape((900))
'''
# # print(data[0])
# init_loglike = -4.217142772e+05
# ts = - 2 * (init_loglike - data)
# print(ts[:, 95])

# fig_hist = plt.figure(0, figsize=(7, 15), dpi=150)
# ax_hist = []
# # for i in range(len(log_files)):
# #     ax_hist.append(fig_hist.add_subplot(5, 1, i+1))
# #     ax_hist[i].hist(ts[i], bins=20)

# ts_plot = np.zeros((900))

# for c, v in enumerate(ts_plot):
#     try:
#         ts_plot[c] = ts[c, 95]
#     except IndexError:
#         pass
# print(ts_plot)
'''Do the actual color map of TS values here
Let's construct a figure first, get the axis limits and all that jazz set
Use reshape to get from a list of 0-899 back again to g, m.
generate list of length 900 with plottable TS values, reshape to (30, 30)
and plot that array. Then set axis limits etc.'''

#ts_plot = ts_plot.reshape((30, 30))
#ts_plot[29, 29] = -200
#print(ts_plot)
# ts_plot = np.random.uniform(low=-210, high=210, size=(30, 30))
fig = plt.figure(1, figsize=(10, 7), dpi=150)
cmap = plt.get_cmap('seismic')
levels = MaxNLocator(nbins=cmap.N).tick_values(-40, 40)
ax = fig.add_subplot(111)
norm = BoundaryNorm(levels, cmap.N)
#ax.set_xlim(1e-1, 1e2)
#ax.set_ylim(1e-1, 1e2)


x_helper = np.linspace(-1, 2, num=30, endpoint=True)
dx_2 = (x_helper[1] - x_helper[0]) / 2
x = np.logspace(-1 - dx_2, 2 + dx_2, num=31, endpoint=True, base=10.0)
y = np.logspace(-1 - dx_2, 2 + dx_2, num=31, endpoint=True, base=10.0)
g_space = np.logspace(-1. - dx_2, 2. - dx_2, num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
m_space = np.logspace(-1. - dx_2, 2. - dx_2, num=30, base=10.0, endpoint=True)    # in neV
grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
for i in range(g_space.shape[0]):
    for j in range(m_space.shape[0]):
        grid[i, j, :] = g_space[i], m_space[j]
grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
xv, yv = np.meshgrid(m_space, g_space)
#print(xv)
xv = xv.reshape((900))
yv = yv.reshape((900))
xmin, xmax, ymin, ymax = 0.3, 30.0, 0.3, 7.0
#aspect = image.shape[0] / image.shape[1] * (xmax - xmin) / (ymax - ymin)

pcol = ax.pcolor(x, y, likes, cmap=cmap, norm=norm, alpha=1)
fig.colorbar(pcol, ax=ax, extend='neither', ticks=(-80, -40, 0, 40, 80), label='TS')

# for c, v in enumerate(np.linspace(0, 900, num=900, endpoint=False, dtype=int).astype(str)):
#     ax.text(xv[c], yv[c], v, fontsize=5, color='red')
ax.set_xlim((np.power(10, -1 - dx_2), np.power(10, 2+dx_2)))
ax.set_ylim((np.power(10, -1 - dx_2), np.power(10, 2+dx_2)))
ax.set_xlabel('$m_{a}$ [neV]')
ax.set_ylabel('$g_{a\gamma\gamma}$ [$10^{-11}$ GeV$^{-1}$]')
#ax.set_xlim((9e-2, 1.2e2))
ax.set_xticks((1e-1, 1e0, 1e1, 1e2))
#ax2 = ax.twinx()
#ax2.set_xlim((-1 - dx_2, 2+dx_2))
#ax2.set_ylim((-1 - dx_2, 2+dx_2))
#ax2.set_ylim((0.08877197088985865, 112.64816923358867))
#ax2.set_xticks((1e-1, 1, 1e1, 1e2))
#ax2.set_yticks((1e-1, 1, 1e1, 1e2))
ax.set_xscale('log')
ax.set_yscale('log')
#im = ax2.imshow(image, extent=[xmin, xmax, ymin, ymax])
plt.savefig(f'colormaps/roi_{which_roi}_color_map.png', dpi=150)
