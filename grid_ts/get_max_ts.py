import numpy as np
import sys
import os

cwd = sys.path[0]

search_path = f'{cwd}/outdata'
roi_dirs = os.listdir(search_path)
# print(roi_dirs)
ts_max_values = np.zeros((100))
cou = 0
for path in roi_dirs:
    roi_path = f'{search_path}/{path}/ts_95'
    if os.path.isdir(roi_path) and 'roi_' in path:
        # do stuff with actual dirs
        print(roi_path)
        ts_files = os.listdir(roi_path)
        ts_values = np.zeros((900))
        for c, fi in enumerate(ts_files):
            # print(f'{roi_path}/{fi}')
            dat = np.loadtxt(f'{roi_path}/{fi}')
            # name = fi.rstrip('.dat')
            # name_list = name.rsplit('_')
            # num = int(name_list[-1])
            # print(num)
            # probably don't need to remember which pixel has the highest TS
            ts_values[c] = dat
        ts_values = np.sort(ts_values)
        ts_max_values[cou] = ts_values[-1]      # take largest TS value per PE
        cou += 1
        if cou > 5:
            break
        else:
            continue
    else:
        continue
print(ts_max_values)
