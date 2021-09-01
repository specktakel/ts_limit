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

simulation = False
try:
    which_roi = int(sys.argv[2])
except:
    which_roi = False

start = int(sys.argv[1])

try:
    print(sys.argv[3])
    rerun = True
except:
    rerun = False

if which_roi:
    roi_file_name = f'sim_{which_roi}.npy'
    roi_name = f'roi_{which_roi}'
else:
    roi_file_name = 'fit_newminuit.npy'
    roi_name = 'fits_01'


#not sure if this is actually correct
if simulation:
    roi_location = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{which_roi}'
else:
    roi_location = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{which_roi}'
    # roi_name = roi_location.rsplit('/', 1)[-1]
    if which_roi:
        outdir = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/jansson12c/roi_{which_roi}'
    else:
        outdir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/jansson12c/orig_data'
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
if not rerun:
    param_range = [i for i in range(start*100, (start+1)*100)]
else:
    param_range = []
    for i in range(900):
        try:
            np.loadtxt(f'{outdir}/gm_{i:03}.dat')
        except:
            param_range.append(i)
print(param_range)
# sys.exit()
prob_dir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/structured_mod_g/b0_83'
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
    prob = np.loadtxt(f'{prob_dir}/gm_{i}_jansson12c.dat')
    # or load probs module at the beginning and generate on the fly
    print(prob.shape)
    j.p = prob #['''some indexing probably required''']
    j.fit()
    j.prepare_outdata()
    # saving procedure here
    
    np.savetxt(f'{outdir}/gm_{i:03}.dat', j.outdata, fmt='%5.4f') 
    if not which_roi:
        best_dnde = j.gta.get_source_dnde(j.name)
        np.savetxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/jansson12c_dnde/gm_{i:03}.dat', best_dnde)
    if time.time() - start_time > 10500:
        break
    else:
        pass

