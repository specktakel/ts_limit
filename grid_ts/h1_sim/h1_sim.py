import numpy as np
from fermipy.gtanalysis import GTAnalysis
import yaml
from iminuit import Minuit
import os
import sys

'''Some used arrays and values.'''
nsim = 100
like_h1 = np.zeros((nsim))    # loglikes of simulated spectrum with alps (of another random bfield!)
like_h0 = np.zeros((1))    # loglike of simulated spectrum w/o alps
like_roi = np.zeros((1))

log_pars = {'norm': {'value':3.35, 'min': 0, 'max': 100, 'error': 0.004, 'scale': 1e-11, 'free': 1},\
            'alpha': {'value':2.070, 'min': -5, 'max': 5, 'error': 0.005, 'scale': 1.0, 'free': 1},\
            'beta': {'value':0.62, 'min': -5, 'max': 10, 'error': 0.027, 'scale': 0.1, 'free': 1},\
            'Eb':{'value': 1.065305815, 'min': 0.01, 'max': 100, 'scale': 1000, 'free': 0}}

alpha = log_pars['alpha']['value']
beta = log_pars['beta']['value']
norm = log_pars['norm']['value']
Eb = log_pars['Eb']['value'] * log_pars['Eb']['scale']
alpha_err = log_pars['alpha']['error']
beta_err = log_pars['beta']['error']
norm_err = log_pars['norm']['error']


path_to_conda = os.environ['CONDA_PREFIX']
cwd = os.getcwd()

photon_path = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/probs/prob645.dat'
outpath = '/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/h1_sim/ts_645.dat'

'''Function definitions.'''
def load_roi(roi_file):
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
    
    gta = GTAnalysis.create(roi_file, config=f'{cwd}/config_modified.yaml')
    
    return gta


def fit_routine(gta):
    for k in range(7):
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
    return gta


def simulate_alps_spectrum(gta, prob):

    gta.set_source_spectrum(source, spectrum_type='FileFunction')
    log10MeV, dnde = gta.get_source_dnde(source)
    x = np.power(10, log10MeV)

    def cost_func_alps(norm, alpha, beta):
        dnde = prob * norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()
    def cost_func(norm, alpha, beta):
        dnde = norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()

    m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
               error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
               limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10),  print_level=1)
    res = m.migrad()
    gta.simulate_roi()
    gta.set_source_spectrum(source, spectrum_type='LogParabola', spectrum_pars=log_pars)
    gta = fit_routine(gta)
    like_roi[0] = - gta.like()    # likelihood of pseudo data, analogously to previous sim.
    return gta


def fit_alps_spectrum(gta, prob, j):
    gta.set_source_spectrum(source, spectrum_type='FileFunction')
    log10MeV, dnde = gta.get_source_dnde(source)
    x = np.power(10, log10MeV)
    def cost_func_alps(norm, alpha, beta):
        dnde = prob * norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()
    def cost_func(norm, alpha, beta):
        dnde = norm * 1.0e-11 * np.power((x / Eb) , -(alpha + beta * 1.0e-1 * np.log((x / Eb))))
        gta.set_source_dnde(source, dnde)
        return gta.like()

    m = Minuit(cost_func_alps, errordef=0.5, norm=norm, alpha=alpha, beta=beta,\
               error_norm=norm_err, error_alpha=alpha_err, error_beta=beta_err,\
               limit_norm=(0, 100), limit_alpha=(-5, 5), limit_beta=(-5, 10),  print_level=1)

    res = m.migrad()
    like_h1[j] = - gta.like()
    return gta



# gta = GTAnalysis.create('fits/simulation_base_roi_more_opt.npy', config='config_modified.yaml')
# src_model = gta.get_src_model(source)


prob = np.loadtxt(photon_path)


gta = load_roi(roi_file)
gta = simulate_alps_spectrum(gta, prob[i])    # maybe pre-generate some ROI with new bfields. use "old" bfields for fitting
gta.set_source_spectrum(source, spectrum_type='LogParabola', spectrum_pars=log_pars)
gta = fit_routine(gta)

for j in range(nsim):
    gta = fit_alps_spectrum(gta, prob[j], j)

TS = 2 * (like_h1 - like_roi)

np.savetxt(outpath, TS)
    






'''
load roi
take one probability with new bfield (unknown, not used yet, like real physical bfield in perseus A),
  put that with filefunction as ngc1275 spectrum
{
simulate roi
go back to logparabola
do fit stuff
    {
    change to file function with usual bfields/photon survival probs and fit w/ minuit
    compare loglikes/TS
    } x 100?
} x ??
'''
    



