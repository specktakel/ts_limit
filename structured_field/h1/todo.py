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


gm = int(sys.argv[1])

# copy analysis dir (roi_simulation/roi_files/fits_01)

### do some path shenanigens here
roi_location = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/h1/fits_01'
roi_name = roi_location.rsplit('/', 1)[-1]
roi_file_name = 'fit_newminuit.npy'
roi_destination = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/h1/fits_01'
outdir = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/outdata/GMF_pshirkov_BSS_PA_-147'
dnde_path = 'dnde_gm_210.dat'
prob_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/structured/no_ebl_gmf_pshirkov/gm{gm}_theta_225_B0_8.3_PA_-147.dat'
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


# get probability of g/m from somewhere with best-fitting logparabola pars
# TODO: get exactly that done, also choose some g/m for testing
dnde = np.loadtxt(dnde_path)
p = np.loadtxt(prob_path)
print(dnde.shape)
print(p.shape)
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

nsim = 2
outdata = np.zeros((nsim, 2))


for i in range(nsim):
    # call gta.simulate_roi()
    j.gta.set_source_dnde(j.name, dnde)
    j.gta.simulate_roi()
    
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
    
    no_alps_ll = - j.gta.like()
    # do fitting with alps, switching to dnde again
    # save loglike
    j.p = np.loadtxt(prob_path)
    j.fit()
    with_alps_ll = - j.gta.like()
    j.gta.simulate_roi(restore=True)
    # repeat
    outdata[i] = np.array([no_alps_ll, with_alps_ll])
    np.savetxt('h1_simulation.dat', outdata, fmt='%5.4f')
