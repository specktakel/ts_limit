from ts_limit.grid import Grid
import numpy as np
import sys
import matplotlib as mpl
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
    print('you should provide an argument')
    sys.exit()
try:
    H0 = np.loadtxt(f'../../roi_simulation/roi_files/roi_{which_roi}/loglike_H0.dat')
except IOError:
    pass

directory = f'roi_{which_roi}'


def check_files(directory, which_roi, prefix='out_', postfix='.dat'):
    '''Checks files in directory for completeness, i.e. if all data lines are present.
    TODO: Include check if all 900 files are present.
    Returns:
    data_set, dictionary with likes, missing lines and missing gm.
    '''
    file_list = listdir(directory)
    data_list = []
    for c, v in enumerate(file_list):
        if prefix in v and postfix in v and not 'copy' in v:
            data_list.append(f'{directory}/{v}')
    
    try:
        loglike_H0 = np.loadtxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{which_roi}/loglike_H0.dat')
    except IOError:
        pass
    numbers = np.zeros((900, 2), dtype=int)
    numbers[:, 0] = np.arange(0, 900, dtype=int)
    missing_lines = []
    likes = np.zeros((900))    # indices: files, simulation, value
    for n in numbers:
        # print(n)
        try:
            temp = np.loadtxt(f'{directory}/{prefix}{n[0]:03}{postfix}')
            if np.any(np.isclose(temp, np.zeros(temp.shape))):
                print(f"{n[0]} has zeros!")
                counter = 0
                for c, line in enumerate(temp):
                    if np.any(np.isclose(line, np.zeros(line.shape))):
                        missing_lines.append([n[0], c])
                        counter += 1
                    else:
                        pass
                numbers[n[0], 1] = counter
            elif temp.shape[0] != 100 or temp.shape[1] != 3:
                raise IOError
            else:
                try:
                    ts_fit = np.sort(2 * (temp[:, -1] - loglike_H0))
                except NameError:
                    ts_fit = np.sort(2 * (temp[:, -1] - temp[:, 0]))
                likes[n[0]] = ts_fit[94]
        except IOError:
            print(f'file {n} is missing')
            missing_lines.append([n[0], 0])
            missing_lines.append([n[0], 25])
            missing_lines.append([n[0], 50])
            missing_lines.append([n[0], 75]) 
    missing_gm = []
    for line in missing_lines:
        arg = [line[0], which_roi, floor(line[1] / 25)]
        if arg not in missing_gm:
            missing_gm.append(arg)
    np.savetxt(f'{directory}/missing_args.txt', np.array(missing_gm), fmt="%1.1i")
    data_set = {'likes': likes, 'missing_lines': missing_lines, 'missing_gm': missing_gm}
    return data_set


def write_ts(likes, which_roi):
    try:
        ts = np.loadtxt('ts/ts.dat')
        ts = ts.reshape((int(ts.flatten().shape[0] / 2), 2))
        for line in ts:
            if which_roi == line[0]:
                break
        else:
            ts = np.append(ts, np.array([[which_roi, np.max(likes)]]), axis=0)
           
    except IOError:
        ts = np.array([[which_roi, np.max(likes)]])
        ts = ts.reshape(1, 2)
    np.savetxt('ts/ts.dat', ts, fmt="%1.1i %1.3f")


def plot_data(likes, which_roi, local_save_dir, afs_save_dir):
    likes = likes.reshape(30, 30)
    fig = plt.figure(1, figsize=(10, 7), dpi=150)
    cmap = plt.get_cmap('seismic')
    levels = MaxNLocator(nbins=cmap.N).tick_values(-40, 40)
    ax = fig.add_subplot(111)
    norm = BoundaryNorm(levels, cmap.N)
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
    xmin, xmax, ymin, ymax = 0.3, 30.0, 0.3, 7.0
    pcol = ax.pcolor(x, y, likes, cmap=cmap, norm=norm, alpha=1)
    fig.colorbar(pcol, ax=ax, extend='neither', ticks=(-40, -20, -10, -5, 0, 5, 10, 20, 40), \
                 label='TS')
    ax.set_xlim((np.power(10, -1 - dx_2), np.power(10, 2+dx_2)))
    ax.set_ylim((np.power(10, -1 - dx_2), np.power(10, 2+dx_2)))
    ax.set_xlabel('$m_{a}$ [neV]')
    ax.set_ylabel('$g_{a\gamma\gamma}$ [$10^{-11}$ GeV$^{-1}$]')
    ax.set_xticks((1e-1, 1e0, 1e1, 1e2))
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.subplots_adjust(hspace=0)
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=2, h_pad=1.5, w_pad=2)
    fig.savefig(f'/afs/desy.de/user/k/kuhlmjul/NGC_1275/maps/ts_fixed_{which_roi}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'colormaps/ts_fixed_{which_roi}.png', dpi=150, bbox_inches='tight')
#     fig.savefig('test_fig.png')


data = check_files(f'roi_{which_roi}', which_roi)
if data['missing_lines'] == []:
    plot_data(data['likes'], which_roi, f'colormaps/ts_fixed_{which_roi}.png', f'/afs/desy.de/user/k/kuhlmjul/NGC_1275/maps/ts_fixed_{which_roi}.png')
    write_ts(data['likes'], which_roi)
else:
    pass
