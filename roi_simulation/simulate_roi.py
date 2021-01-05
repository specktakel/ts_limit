import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
import os
import sys
import matplotlib.pyplot as plt
# import time
cwd = os.getcwd()
# t_0 = int(time.time())
input_arg = sys.argv[1]
input_num = int(input_arg)
nsim = 2
sim_list = [input_num * nsim + l + 105 for l in range(nsim)]
# sim_list.append(input_num)
path_to_conda = os.environ['CONDA_PREFIX']
print(sim_list)
print(cwd)

with open(cwd+'/config_local.yaml', 'r') as i:
    config = yaml.safe_load(i)

config['fileio']['workdir'] =  cwd+'/fits'
config['fileio']['outdir'] = cwd+'/fits'
config['fileio']['logfile'] = cwd+'/fits/fermipy.log'
config['data']['ltcube'] = cwd+'/fits/ltcube_00.fits'
config['model']['galdiff'] = path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
config['model']['isodiff'] = path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
config['logging']['verbosity'] = 4

source = config['selection']['target']

with open(cwd+'/config_modified.yaml', 'w') as o:
    yaml.dump(config, o)
sys.exit()
gta = GTAnalysis.create(cwd+'/fits/fully_optimized.npy', config=cwd+'/config_modified.yaml')
base_like = gta.like()
# simulated_like = np.zeros(nsim)
# t_1 = int(time.time())
# print(f'time for loading base roi: {t_1 - t_0}')
for i in sim_list:
    print(f'doing sim with index {i}')
    like_save_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{i}/loglike_H0_{i}.dat'     # loglike of null hypothesis (no alps)
    roi_dir = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{i}'
    gta.free_sources(free=False)
    gta.simulate_roi()
    for j in range(5):
        gta.optimize()
    gta.free_source('isodiff')
    gta.free_source('galdiff')
    gta.free_source(source)
    gta.free_sources(distance=3, pars='norm')
    gta.free_sources(minmax_ts=[100, None], pars='norm')
    fit_result = gta.fit(optimizer='NEWMINUIT', reoptimize=True)
    like = gta.like()
    like = np.array([like])
    # simulated_like[i] = like
    try:
        os.mkdir(roi_dir)
        print(f'dir {roi_dir} created succesfully') 
    except FileExistsError:
        print(f'dir {roi_dir} already exists, proceeding...')
    
    gta.write_roi(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{i}/roi_{i}.npy')
    np.savetxt(like_save_path, like)
    gta.simulate_roi(restore=True)
    gta.load_roi('fully_optimized')
# t_2 = int(time.time())
# print(f'time for simulations: {t_2 - t_1}')
# np.savetxt(like_save_path, simulated_like, header=f'base_loglike: {base_like}')

# print(f'total elapsed time (in seconds): {t_2 - t_0}, only for generation and fitting: {t_2 - t_1}')
# print(f'approx time per 100 simulations: {int((t_2 - t_1) / nsim * 100)}')

