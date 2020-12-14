import yaml
from fermipy.gtanalysis import GTAnalysis
import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
from iminuit import Minuit
import os
import sys

path_to_conda = os.environ['CONDA_PREFIX']
cwd = os.getcwd()
source = '4FGL J0319.8+4130'

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
likelihoods = np.zeros((5))
gta = GTAnalysis(config='config_modified.yaml')
gta.setup()
model = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}

for i in range(1,6):
    gta.optimize()
    gta.free_sources(free=False)
    gta.free_source(source)
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_sources(distance=3, pars='norm')
    gta.free_sources(minmax_ts=[100, None], pars='norm')
    gta.fit(optimizer='NEWMINUIT', reoptimize=True)
    maps = gta.residmap(f'../maps/opt_alternating{i}', model=model, make_plots=True)
    maps = gta.tsmap(f'../maps/opt_alternating_{i}', model=model, make_plots=True)
    gta.write_roi(f'opt_{i}', make_plots=True)
    likelihoods[i-1] = - gta.like()


np.savetxt('optimization_process_likes_alternating.dat', likelihoods)
