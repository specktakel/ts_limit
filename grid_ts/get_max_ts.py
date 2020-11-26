import numpy as np
import sys
import os

cwd = sys.path[0]

search_path = f'{cwd}/outdata'
# roi_dirs = os.listdir(search_path)
# print(roi_dirs)
ts_file_path = 'ts_max_values_wip.dat'
try:
    ts_max_values = np.loadtxt(ts_file_path)
    print('ts file found')
except:
    ts_max_values = np.zeros((100))
    print('no ts file found, creating empty array')
cou = 0
for c, val in enumerate(ts_max_values):
    print(c)
    if val == 0:
        roi_path = f'{search_path}/roi_{c}/ts_95'
    # isolate number of ROI
    # if os.path.isdir(roi_path) and 'roi_' in path:
    #     # do stuff with actual dirs
    #     roi_num = int(path.split('_')[-1])
    #     print(roi_num)
    #     print(roi_path)
    #     if ts_max_values[roi_num] == 0:
    #         print("entry non zero, continuing with next roi")
    #         continue
        ts_files = os.listdir(roi_path)
        if len(ts_files) == 900:
            ts_values = np.zeros((900))
            for co, fi in enumerate(ts_files):
            # print(f'{roi_path}/{fi}')
                dat = np.loadtxt(f'{roi_path}/{fi}')
            # name = fi.rstrip('.dat')
            # name_list = name.rsplit('_')
            # num = int(name_list[-1])
            # print(num)
            # probably don't need to remember which pixel has the highest TS
                ts_values[co] = dat
            ts_values = np.sort(ts_values)
            ts_max_values[c] = ts_values[-1]      # take largest TS value per PE
        cou += 1
        # if cou > 1:
        #     break
        # else:
        #    continue
    else:
        continue
print(ts_max_values)
np.savetxt(ts_file_path, ts_max_values)
