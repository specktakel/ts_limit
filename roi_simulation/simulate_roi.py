import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
import os
import sys
import matplotlib.pyplot as plt
import shutil
cwd = os.getcwd()
input_arg = sys.argv[1]
input_num = int(input_arg)
# nsim = 2
path_to_conda = os.environ['CONDA_PREFIX']
print(cwd)
master_loc = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/fits_01'
with open(cwd+'/config_local.yaml', 'r') as i:
    config = yaml.safe_load(i)

if 'n1/kuhlmann' in cwd:
    workdir = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{input_arg}'
else:
    workdir = f'{cwd}/fits_01'

config['fileio']['workdir'] =  f'{workdir}'
config['fileio']['outdir'] = f'{workdir}'
config['fileio']['logfile'] = f'{workdir}/fermipy.log'
config['data']['ltcube'] = f'{workdir}/ltcube_00.fits'
config['model']['galdiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
config['model']['isodiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
config['logging']['verbosity'] = 4
source = config['selection']['target']

with open(cwd+'/config_modified.yaml', 'w') as o:
    yaml.dump(config, o)


# sys.exit()
if 'n1/kuhlmann' in cwd:
    try:
        os.mkdir(workdir)
    except FileExistsError:
        print('output directory already exists')
    roi_path = workdir
    print(roi_path)
    files = os.listdir(master_loc)
    print('files at master loc:', files)
    for f in files:
        shutil.copy2(f'{master_loc}/{f}', f'{workdir}')    # copying from master file location
else:
    roi_path = f'{cwd}/fits_01'

gta = GTAnalysis.create(f'{roi_path}/fit_newminuit', cwd+'/config_modified.yaml')
print('loaded loglike:', -gta.like())
roi_npy = np.load(f'{roi_path}/fit_newminuit.npy', allow_pickle=True).flat[0]
print('numpy loglike:', roi_npy['roi']['loglike'])
like_save_path = f'{workdir}/loglike_H0_{input_arg}.dat'     # loglike of null hypothesis (no alps)
gta.free_sources(free=False)
gta.simulate_roi()
for j in range(5):
    gta.optimize()
print("loglike of saved roi", -gta.like())
gta.write_roi(f'sim_{input_arg}')
gta.free_source(source)
gta.fit(optimizer='NEWMINUIT', reoptimize=True)
gta.free_source(source, free=False)
gta.free_source(source, pars='norm')
gta.free_source('isodiff')
gta.free_source('galdiff')
gta.fit(optimizer='NEWMINUIT', reoptimize=True)
like = np.array([-gta.like()])
np.savetxt(like_save_path, like)
