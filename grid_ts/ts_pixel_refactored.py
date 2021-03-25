from ts_limit.grid_ts.janitor import Janitor
import numpy as np
import os
import sys
import shutil
import psutil
import time
start_time = time.time()
process = psutil.Process(os.getpid())
nsim = 3


'''Get needed values from CLI.'''
try:
    which_gm = int(sys.argv[1])
except ValueError:
    print('gm not a number, exiting')
    sys.exit()

try:
    which_roi = int(sys.argv[2])
    roi_name = f'roi_{which_roi}'
except ValueError:
    print('roi not a number, use instead str')
    roi_name = sys.argv[2]

which_range = int(sys.argv[3])


'''If rerunning some args, catch this and handle things differently. This concerns, e.g., indices of simulations to be done.'''
try:
    num = int(sys.argv[4])
    rerun = True 
    print('rerunning some args') 
except IndexError:
    rerun = False


'''Set up all necessary paths and directories.'''
path_dict = {}
package_path = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit'
#prob_path subject to deletion
prob_path = f'{package_path}/grid_survival_prob/new_probs/roi_name/prob_{which_gm:03}.dat'
save_dir = f'{package_path}/grid_ts/outdata/{roi_name}'
save_path = f'{save_dir}/out_{which_gm:03}.dat'
cwd = os.getcwd()

'''Need to check if this is running on the cluster or astro-wgs.'''
if not 'n1/kuhlmann' in cwd:
    print('Im running on the cluster')
    roi_dir = f'{cwd}/{roi_name}'
else:
    print('Im running on the wgs')
    roi_dir = f'{package_path}/grid_ts/roi_tempdir'
    roi_origin = f'{package_path}/grid_ts/{roi_name}'

    try:
        os.mkdir(f'{roi_dir}')
    except FileExistsError:
        print('temp directory exists')

    print(f'should copy from {roi_origin} to {roi_dir}')
    files = os.listdir(roi_origin)
    for f in files:
        # shutil.copy2(f'{roi_origin}/{f}', f'{roi_dir}')     # should overwrite any files present, clean up afterwards anyway...
        continue

try:
    roi_file = f'{roi_dir}/sim_{which_roi}.npy'
except NameError:
    roi_file = f'{roi_dir}/edisp_-2.npy'

path_dict['prob_path'] = prob_path
path_dict['save_dir'] = save_dir
path_dict['save_path'] = save_path
path_dict['cwd'] = cwd
path_dict['roi_dir'] = roi_dir
path_dict['roi_file'] = roi_file
path_dict['package_path'] = package_path
path_dict['roi_file'] = roi_file
path_dict['config_path'] = f'{roi_dir}/config_modified.yaml'

'''Create output paths if necessary.'''
try:
    os.mkdir(save_dir)
    print(f'created output path: {save_dir}')
except FileExistsError:
    print(f'output path already exists, continuing...')
print(f'cwd: {cwd}')
print(path_dict)

if rerun:
    try:
        data = np.loadtxt(f"{save_dir}/out_{which_gm:03}.dat")
        indices = []
        if data.shape[0] == 100 and data.shape[1] == 3:
            for c, line in enumerate(data):
                if np.any(np.isclose(line, np.zeros(line.shape))):
                    indices.append(c)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("file has wrong shape or is not found")
        data = np.zeros((100, 3), dtype=float)
        np.savetxt(f"{save_dir}/out_{which_gm:03}.dat", data)
        start = which_range * nsim
        end = (which_range + 1) * nsim
        indices = [i for i in range(start, end)]
else:
    start = which_range * nsim
    end = (which_range + 1) * nsim    
    indices = [i for i in range(start, end)]
print(path_dict)
print(indices)
# sys.exit()
print('memory pre fermipy:', process.memory_info().rss * 1e-6)
obj = Janitor(which_gm, 69, which_range, path_dict)
print('memory post fermipy:', process.memory_info().rss * 1e-6)


for i in indices:
    obj.index = i
    obj.fit()
    print('memory used:', process.memory_info().rss * 1e-6)
    obj.write_outdata()
    '''Check if roughly enough time for the next fit remains.
    Cluster limit is at 3 hours. One fit takes approx. 2-5 minutes.'''
    current_time = time.time()
    if current_time - start_time > 10500:
        sys.exit()
    else:
        pass 
    if i != indices[-1]:
        obj.bootleg_reload()
    else:
        pass

