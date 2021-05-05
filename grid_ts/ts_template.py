import numpy as np
from fermipy.gtanalysis import GTAnalysis
from ts_limit.grid_ts.janitor import Janitor
from ts_limit.grid import Grid
import time
import os
import sys
from shutil import copy2

start_time = time.time()


### do some path shenanigens here
roi_location = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/fits_01'
roi_name = roi_location.rsplit('/', 1)[-1]
roi_file_name = 'fit_newminuit.npy'
roi_destination = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/roi_tempdir'
outdir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata'

# ask for cluster?
cwd = os.getcwd()

if not '/n1/kuhlmann/' in cwd:
    # on cluster, everything copied by bash wrapper
    workdir = f'{os.getcwd()}/{roi_name}'
    roi_file_path = f'{workdir}/{roi_file_name}'
    config_path = f'{workdir}/config.yaml'
    pass
else:
    # on wgs, should copy to roi_tempdir
    files = os.listdir(roi_location)
    workdir = roi_destination
    roi_file_path = f'{roi_destination}/{roi_file_name}'
    config_path = f'{roi_destination}/config.yaml'
    for f in files:
        break
        # copy2(f'{roi_location}/{f}', roi_destination)


### get param_range or similar from command line args
param_range = [0, 1, 2, 3]
prob_dir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/structured'


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
    prob = np.loadtxt(f'{prob_dir}/gm_{i}_theta_225_B0_8.3.dat')
    # or load probs module at the beginning and generate on the fly
    print(prob.shape)
    j.p = prob #['''some indexing probably required''']
    j.fit()
    j.prepare_outdata()
    # saving procedure here
    np.savetxt(f'{outdir}/test_{i}.dat', j.outdata) 
    if time.time() - start_time > 10500:
        break
    else:
        pass
    if i != param_range[-1]:
        j.bootleg_reload()
    else:
        break

