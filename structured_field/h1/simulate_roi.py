#pseudo code as rough guide for simulation and analysis under h1


#imports
import numpy as np
from fermipy.gtanalysis import GTAnalysis
from ts_limit.grid_ts.janitor import Janitor
from ts_limit.grid import Grid
import time
import os
import sys
from shutil import copy2

#roi number int num
gm = int(sys.argv[1])

#index of gm to be simulated under
num = int(sys.argv[2])
# copy analysis dir (roi_simulation/roi_files/fits_01)

### do some path shenanigens here
roi_location = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/fits_01'
roi_name = roi_location.rsplit('/', 1)[-1]
roi_file_name = 'fit_newminuit.npy'
roi_destination = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/h1/roi_tempdir'
outdir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/h1'
dnde_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/jansson12c_dnde/gm_{gm:03}.dat'
# prob_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/structured_mod_g/b0_83/gm_{gm}_jansson12c.dat'
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
    for f in files:
        break
        # copy2(f'{roi_location}/{f}', roi_destination)


# get probability of g/m from somewhere with best-fitting logparabola pars
# TODO: get exactly that done, also choose some g/m for testing
dnde_arr = np.loadtxt(dnde_path)
dnde = dnde_arr[1]
print(dnde.shape)
print(dnde_path)
compare = np.loadtxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/jansson12c_dnde/gm_{gm:03}.dat')[1]
print(np.all(np.isclose(dnde, compare)))
if not np.all(np.isclose(dnde, compare)):
    raise ValueError('dnde wrong')
# set dnde of NGC1275 to that
### define gtanalysis instance
Janitor.write_yaml(config_path, f'{workdir}/config_written.yaml',
                   data={'ltcube': f'{workdir}/ltcube_00.fits'},
                   fileio={'logfile': f'{workdir}/fermipy.log',
                           'outdir': workdir,
                           'workdir': workdir})

# sys.exit()
gta = GTAnalysis.create(roi_file_path, config=f'{workdir}/config_written.yaml')
j = Janitor(gta)


# call gta.simulate_roi()
j.gta.set_source_dnde(j.name, dnde)
e, d = j.gta.get_source_dnde(j.name)
print(d)
worked = np.all(np.isclose(dnde, d))
if not worked:
    raise ValueError('wrong dnde returned.')
j.gta.simulate_roi()
# j.bootleg_reload()
j.set_logparabola()
# do the usual fitting stuff w/o alps
# save loglike
for c in range(5):
    o_dict = j.gta.optimize()
    if np.abs(o_dict['dloglike']) < 1.0:
        print(f'opt {c}, done here')
        break
j.gta.free_sources(free=False)
j.gta.free_source('isodiff')
j.gta.free_source('galdiff')
j.gta.free_sources(distance=3.0, pars='norm')
j.gta.free_sources(minmax_ts=[100, None], pars='norm')
j.gta.free_source(j.name)
j.gta.fit(optimizer='NEWMINUIT', reoptimize=True)
j.gta.write_roi(f'sim_{num}.npy')
loglike_h0 = np.array([j.gta.like()])
np.savetxt('fits_01/loglike_H0.dat', loglike_h0, fmt='%5.4f')
