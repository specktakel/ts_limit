import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
from iminuit import Minuit
import os
import sys

roi = int(sys.argv[1])
# gm = int(sys.argv[2])


'''Some used arrays and values.'''
nsim = 5
like_h1 = np.zeros((nsim))    # loglikes of simulated spectrum with alps (of another random bfield!)
like_h0 = np.zeros((1))    # loglike of simulated spectrum w/o alps
like_roi = np.zeros((1))

log_pars = {'norm': {'value':3.35, 'min': 0, 'max': 100, 'error': 0.004, 'scale': 1e-11, 'free': 1},\
            'alpha': {'value':2.070, 'min': -5, 'max': 5, 'error': 0.005, 'scale': 1.0, 'free': 1},\
            'beta': {'value':0.62, 'min': -5, 'max': 10, 'error': 0.027, 'scale': 0.1, 'free': 1},\
            'Eb':{'value': 1.065305815, 'min': 0.01, 'max': 100, 'scale': 1000, 'free': 0}}
# gm_vals = np.array([849, 435, 446])
gm_vals = []
gm_vals.append(int(sys.argv[2]))
alpha = log_pars['alpha']['value']
beta = log_pars['beta']['value']
norm = log_pars['norm']['value']
Eb = log_pars['Eb']['value'] * log_pars['Eb']['scale']
alpha_err = log_pars['alpha']['error']
beta_err = log_pars['beta']['error']
norm_err = log_pars['norm']['error']


path_to_conda = os.environ['CONDA_PREFIX']
cwd = os.getcwd()
source = '4FGL J0319.8+4130'


photon_base_path = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/probs'
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

if roi == 0:
    roi_file = 'fully_optimized.npy'

elif roi == 1:
    roi_file = 'simulation_base_roi.npy'

gta = GTAnalysis.create(f'fits/{roi_file}', config='config_modified.yaml')
init_like = - gta.like()
src_model = gta.get_src_model(source)
norm = src_model['param_values'][0] * 1.0e11
alpha = src_model['param_values'][1]
beta = src_model['param_values'][2] * 1.0e1
norm_err = src_model['param_errors'][0] * 1.0e11
alpha_err = src_model['param_errors'][1]
beta_err = src_model['param_errors'][2] * 1.0e1
Eb = src_model['param_values'][3]
gta.set_source_spectrum(source, spectrum_type='FileFunction')
log10MeV, dnde = gta.get_source_dnde(source)
x = np.power(10, log10MeV)
loglikes = np.zeros((nsim, 5))
for i in gm_vals:
    prob = np.loadtxt(f'{photon_base_path}/prob{i:03.0f}.dat')
    for j in range(nsim):
        loglikes[j, 0] = - gta.like()
        p = prob[j]
        def cost_func_alps(norm, alpha, beta):
            dnde = p * norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
            gta.set_source_dnde(source, dnde)
            return gta.like()
        def cost_func(norm, alpha, beta):
            dnde = norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
            gta.set_source_dnde(source, dnde)
            return gta.like()
    
    
        m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
                   error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
                   limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10),  print_level=1)
        r = m.migrad()
        loglikes[j, 1] = -gta.like()
        gta.fit(opimizer='NEWMINUIT', reoptimize=True)
        loglikes[j, 2] = -gta.like()
        m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
                   error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
                   limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10),  print_level=1)
        r = m.migrad()
        loglikes[j, 3] = -gta.like()
        gta.fit(opimizer='NEWMINUIT', reoptimize=True)
        loglikes[j, 4] = -gta.like()
        try:
            gta.load_roi(roi_file)
        except:
            gta.load_roi(f'fits/{roi_file}')
        gta.set_source_spectrum(source, spectrum_type='FileFunction')
    np.savetxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/fit_testing/fitting_strategy_{roi}_{i:03.0f}.dat', np.transpose(loglikes))

