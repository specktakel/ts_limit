import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
from iminuit import Minuit
import os
import sys

path_to_conda = os.environ['CONDA_PREFIX']
cwd = os.getcwd()
nsim = 1
sims_per_b = 1
photon_path = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/probs/prob645.dat'
print('do stuff with yaml config')
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
print('done with yaml')



roi_file = 'sim_with_alps'
# load photon survival probability
prob = np.loadtxt(photon_path)
print(f'loaded photon probability: {photon_path}')
# load roi

# gta = GTAnalysis.create('fits/simulation_base_roi_more_opt.npy', config='config_modified.yaml')
# src_model = gta.get_src_model(source)
'''
norm = src_model['param_values'][0] * 1.0e11
alpha = src_model['param_values'][1]
beta = src_model['param_values'][2] * 1.0e1
norm_err = src_model['param_errors'][0] * 1.0e11
alpha_err = src_model['param_errors'][1]
beta_err = src_model['param_errors'][2] * 1.0e1
Eb = src_model['param_values'][3]
'''

like_h1 = np.zeros((nsim, sims_per_b))
like_h0 = np.zeros((nsim, sims_per_b))
like_roi = np.zeros((nsim))

log_pars = {'norm': {'value':3.35, 'min': 0, 'max': 100, 'error': 0.004, 'scale': 1e-11, 'free': 1},\
         'alpha': {'value':2.070, 'min': -5, 'max': 5, 'error': 0.005, 'scale': 1.0, 'free': 1},\
         'beta': {'value':0.62, 'min': -5, 'max': 10, 'error': 0.027, 'scale': 0.1, 'free': 1},\
         'Eb':{'value': 1.065305815, 'min': 0.01, 'max': 100, 'scale': 1000, 'free': 0}}
gta = GTAnalysis.create('fits/simulation_base_roi.npy', config='config_modified.yaml')
print('pre-anything likelihood:', gta.like())
for i in range(nsim):
    alpha = log_pars['alpha']['value']
    beta = log_pars['beta']['value']
    norm = log_pars['norm']['value']
    Eb = log_pars['Eb']['value']
    alpha_err = log_pars['alpha']['error']
    beta_err = log_pars['beta']['error']
    norm_err = log_pars['norm']['error']
    # take pre-fitted ROI and put alps-modified spectrum in NGC1275.
    gta.set_source_spectrum(source, spectrum_type='FileFunction')
    log10MeV, dnde = gta.get_source_dnde(source)
    x = np.power(10, log10MeV)
    def cost_func_alps(norm, alpha, beta):
        dnde = prob[i] * norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()
    def cost_func(norm, alpha, beta):
        dnde = norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()
    input('any key to continue')
    m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
               error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
               limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10),  print_level=1)
    res = m.migrad()
    print('likelihood with alps-spectrum:', gta.like())
    input('any key to continue')
    like_roi[i] = - gta.like()
    gta.write_roi(roi_file)
    print('roi file is written, doing simulation and fitting')
    for j in range(sims_per_b):
        # simulate under alps hypothesis H1
        gta.simulate_roi()
        # set source spectrum to logparabola w/o alps, fit and get loglike
        gta.set_source_spectrum(source, spectrum_type='LogParabola', spectrum_pars=log_pars)
        # do stuff here
        for k in range(1):
            o = gta.optimize()
            print(o['dloglike'])
            if o['dloglike'] < 0.8:
                break
        gta.free_sources(free=False)
        gta.free_source(source)
        gta.free_source('galdiff')
        gta.free_source('isodiff')
        gta.free_sources(distance=3, pars='norm')
        gta.free_sources(minmax_ts=[100, None], pars='norm')
        gta.fit(opimizer='NEWMINUIT', reoptimize=True)
        print('likelihood with pure logparabola:', gta.like())
        input('any key')
        like_h0[i, j] = - gta.like()
        gta.set_source_spectrum(source, spectrum_type='FileFunction')
        # this part looks dodgy...
        '''
        for k in range(5):
             o = gta.optimize()
            print(o['dloglike'])
            if o['dloglike'] < 0.8:
                break
        gta.free_sources(free=False)
        gta.free_source(source)
        gta.free_source('galdiff')
        gta.free_source('isodiff')
        gta.free_sources(distance=3, pars='norm')
        gta.free_sources(minmax_ts=[100, None], pars='norm')
        gta.fit(opimizer='NEWMINUIT', reoptimize=True)
        '''
        m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
                   error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
                   limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10), print_level=1)
        res = m.migrad()
        like_h1[i, j] = - gta.like()
        gta.simulate_roi(restore=True)
        # gta.load_roi(roi_file)

ts = 2 * (like_h1 - like_h0)
np.savetxt('like_logpar.dat', like_h0)
np.savetxt('like_alps.dat', like_h1)
np.savetxt('ts.dat', ts)
np.savetxt('like_roi.dat', like_roi)


'''
pseudo code for mental stability here:

load probs
setup gta object
for i in range(nsim):
    p = prob(i)
    minuit fitting of logparabola * p
    like_1 = gta.like()
    for j in range(sims_per_bfield):
        inject poisson-mc ccube
        gta.optimize() times 6
        free source stuff
        gta.fit()
        like_2 = gta.like()
        do ts
'''
    



