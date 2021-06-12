import numpy as np
from fermipy.gtanalysis import GTAnalysis
from ts_limit.grid_ts.janitor import Janitor
from ts_limit.grid import Grid
import time
import os
import sys
from shutil import copy2
# start = int(sys.argv[1])
start_time = time.time()

simulation = True
which_roi = int(sys.argv[1])
if simulation:
    roi_file_name = f'sim_{which_roi}.npy'
    roi_location = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{which_roi}'
    roi_name = roi_location.rsplit('/', 1)[-1]
else:
    roi_location = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/fits_01'
    roi_name = roi_location.rsplit('/', 1)[-1]
    roi_file_name = 'fit_newminuit.npy'
    outdir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/GMF_pshirkov_BSS_PA_-147'

roi_destination = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/roi_tempdir'


# ask for cluster?
cwd = os.getcwd()

if not '/n1/kuhlmann/' in cwd:
    # on cluster, everything copied by bash wrapper
    workdir = f'{os.getcwd()}/{roi_name}'
    roi_file_path = f'{workdir}/{roi_file_name}'
    config_path = f'{workdir}/config.yaml'
else:
    # on wgs, should copy to roi_tempdir
    files = os.listdir(roi_location)
    workdir = roi_destination
    roi_file_path = f'{roi_destination}/{roi_file_name}'
    config_path = f'{roi_destination}/config.yaml'
    try:
        os.mkdir(roi_destination)
    except FileExistsError:
        print('dir already exists')
    for f in files:
        break
        # copy2(f'{roi_location}/{f}', roi_destination)

# sys.exit()
### get param_range or similar from command line args
param_range = [i for i in range(650, 655)]
prob_dir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/structured/no_ebl_gmf_pshirkov/'
print(param_range)
##  sys.exit()

### define gtanalysis instance
Janitor.write_yaml(config_path, f'{workdir}/config_written.yaml',
                   data={'ltcube': f'{workdir}/ltcube_00.fits'}, 
                   fileio={'logfile': f'{workdir}/fermipy.log',
                           'outdir': workdir,
                           'workdir': workdir})



# sys.exit()
gta = GTAnalysis.create(roi_file_path, config=f'{workdir}/config_written.yaml')

### init janitor object

j = Janitor(gta)


### actual fitting in loop over param_range

for i in param_range:
    prob = np.loadtxt(f'{prob_dir}/gm{i}_theta_225_B0_8.3_PA_-147.dat')
    # or load probs module at the beginning and generate on the fly
    print(prob.shape)
    j.p = prob #['''some indexing probably required''']
    j.fit()
    j.prepare_outdata()
    # saving procedure here
    np.savetxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/roi_{which_roi}/gm_{i:03}.dat', j.outdata, fmt='%5.4f') 
    if time.time() - start_time > 10500:
        break
    else:
        pass
    if i != param_range[-1]:
        j.bootleg_reload()
        print(j.gta.like())
        print(j.init_like)
    else:
        break

