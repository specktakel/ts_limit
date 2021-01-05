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
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from math import floor
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from os import listdir
from os.path import isfile
from matplotlib import rc
import sys
import re


try:
    which_roi = int(sys.argv[1])
except IndexError:
    pass

# which_roi = int(sys.argv[1])

# directory = f'roi_{which_roi}/ts_95'
directory = f'roi_{which_roi}'
# directory = 'orig_data/fixed'
prefix = 'out_'
postfix = '.dat'

file_list = listdir(directory)
#print(file_list)
data_list = []
for c, v in enumerate(file_list):
    if prefix in v and postfix in v and not 'copy' in v:
        data_list.append(f'{directory}/{v}')


numbers = np.zeros((900, 2), dtype=int)
numbers[:, 0] = np.arange(0, 900, dtype=int)
missing_lines = []
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

for v in data_list:
    if 'free_source_test' not in v and 'copy' not in v:
        name = v.rsplit('_', maxsplit=1)
        num = int(name[1].rstrip('.dat'))
        # print(nium)
        numbers[num, 1] = 1
        temp = np.loadtxt(v)
        # print(temp.shape)
        if np.any(np.isclose(temp, np.zeros(temp.shape))):
            print(f"{v} has zeros!")
            for c, line in enumerate(temp):
                if np.any(np.isclose(line, np.zeros(line.shape))):
                    # print(c)
                    missing_lines.append([num, c])
                else:
                    pass
        elif temp.shape[0] != 100:
            
            print(v, "wrong shape")
            missing_lines.append([num, 0])
            missing_lines.append([num, 25])
            missing_lines.append([num, 50])
            missing_lines.append([num, 75])
            continue 
        else:
            pass
        ts_fit = 2 * np.sort(temp[:, -1] - temp[:, 0])
         #print(ts_fit.shape)
        # print(v, ts.shape)
        likes[num] = ts_fit[94]
        # ts_minuit = 2 * np.sort(temp[:, 1] - temp[:, 0])
        # likes[num] = ts_minuit[94]
        # likes[num] = temp[0] - temp[1]
        # break
# np.savetxt('orig_updated_lite.dat', likes)    
# likes = likes.reshape((30, 30))

print(np.argwhere(numbers[:, 1]==0))
print(missing_lines)
missing_gm = []
for line in missing_lines:
    arg = [line[0], which_roi, floor(line[1] / 25)]
    if arg not in missing_gm:
        missing_gm.append(arg)
        
missing_args = np.array(missing_lines)
np.savetxt(f'roi_{which_roi}/missing_args.txt', np.array(missing_gm), fmt="%1.1i")


try:
    ts_values = np.loadtxt('ts_mock_data.dat')
except:
    ts_values = np.zeros((100, 1), dtype=float)

ts_values[which_roi - 100] = np.max(likes)
ts_values = ts_values.reshape((100, 1))
np.savetxt('ts_mock_data.dat', ts_values)
sys.exit()


'''
#directory = 'orig_data/ts_95'
#prefix = 'ts_95_opt_'
postfix = '.dat'
file_list = listdir(directory)
# file_list = listdir(directory)
#print(file_list)
# data_list = []
# for c, v in enumerate(file_list):
#     if prefix in v and postfix in v:
#         data_list.append(v)
#print('data list for not-optimised fit:', data_list)


for fi in data_list:
    name = fi.rsplit('_', maxsplit=1)
    # print(name)
    num = int(name[1].rstrip('.dat'))
    # print(num)
    temp = np.loadtxt(f'{fi}')
    ts = 2 * np.sort(temp[:, 2] - temp[:, 0])
    # ts_95 = temp
    likes[num] = ts[94]
    #break
'''
# sys.exit()
likes = likes.reshape((30, 30))
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
# fig.suptitle('Improvement of updated fitting strategy')
pcol = ax.pcolor(x, y, likes, cmap=cmap, norm=norm, alpha=1)
fig.colorbar(pcol, ax=ax, extend='neither', ticks=(-40, -20, -10, -5, 0, 5, 10, 20, 40), \
             label='TS')
# print(np.max(likes))
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
fig.subplots_adjust(hspace=0)
fig.patch.set_facecolor('white')
fig.tight_layout(pad=2, h_pad=1.5, w_pad=2)
#fig.subplots_adjust(hspace=0.5)
fig.savefig(f'/afs/desy.de/user/k/kuhlmjul/NGC_1275/maps/ts_fixed_{which_roi}.png', dpi=150, bbox_inches='tight')
fig.savefig(f'colormaps/ts_fixed_{which_roi}.png', dpi=150, bbox_inches='tight')
