import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from ebltable.tau_from_model import OptDepth
import numpy as np
from fermipy.gtanalysis import GTAnalysis
import astropy.io.fits as pyfits
from gammaALPs import Source, ALP, ModuleList
from gammaALPs.base import environs, transfer
from iminuit import Minuit
from shutil import copytree


path_to_conda = os.environ['CONDA_PREFIX']
cwd = os.getcwd()
# %matplotlib inline
which_gm = int(sys.argv[1])    # use this to find correct g, m
which_roi = int(sys.argv[2])
which_range = int(sys.argv[3])

'''Set up roi/parameters space'''
nsim = 25
grid = np.linspace(0, 900, num=900, endpoint=False, dtype=int)
roi_path = f'{cwd}/roi_{which_roi}'
print('roi_path:', roi_path, os.path.isdir(roi_path))


'''For testing, if running on cluster, copy roi files. else: dont.'''

if not 'n1' in cwd:
    print('Im running on the cluster')
    # copytree(roi_path, f'{cwd}')
    roi_path = f'{cwd}/roi_{which_roi}'
else:
    print('Im running on the wgs')
    roi_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_{which_roi}'
    # roi_file = f'{roi_path}/simulation_base_roi_more_opt.npy'
# print('roi_file:', roi_file, os.path.isfile(roi_file))

roi_path = f"{cwd}/fits"      # for testing on astro-wgs1[1, 2]
'''Set up g/m space and get correct combination'''
g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
for i in range(g_space.shape[0]):
    for j in range(m_space.shape[0]):
        grid[i, j, :] = g_space[i], m_space[j]
grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
g = grid[which_gm, 0]
m = grid[which_gm, 1]
# print(grid)
# print('roi_number:', which_roi)
print('g:', g)
print('m:', m)


'''Set up variables and save paths'''
prob_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/new_probs/orig_data_fixed/prob_{which_gm:03}.dat'
# loglike_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/roi_{which_roi}'
# ts_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/orig_data/updated_lite_ts'
loglike_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/orig_data/fixed'
roi_file = f'{roi_path}/fully_optimized.npy'
print('roi_file:', roi_file)
try:
    os.mkdir(loglike_path)
    print(f'created output path: {loglike_path}')
except FileExistsError:
    print(f'output path already exists, continuing...')
print(f'cwd: {cwd}')

''' Do stuff with the yaml config here:'''

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

# print(os.listdir(cwd))
'''class definition'''
class janitor:
    '''Because janitors do the work.
    '''

    def __init__(self, name, gta):
        self.name = name
        self.gta = gta
        # self.init_loglike = np.zeros((nsim), dtype=float)
        self.res = []
        #_ = fit_routine(self)
        self.ts = np.zeros((100), dtype=float)
        self.ts_95 = np.zeros((1), dtype=float)
        self.loglike = np.zeros((100), dtype=float)
        self.outdata = np.zeros((nsim, 3), dtype=float)
        self.ll_minuit = np.zeros((nsim), dtype=float)
        self.ll_fit = np.zeros((nsim), dtype=float)
        # self.existing_data = np.loadtxt(outpath)
    def log_parabola(self, x, norm, alpha, beta):
        '''Log parabola.
        Returns: differential flux dN/dE
        '''

        dnde = norm * 1.0e-11 * np.power((x / self.Eb) , -(alpha + beta * 1.0e-1 * np.log((x / self.Eb))))
        return dnde

    
    def survival_logparabola(self, en, norm, alpha, beta):
        '''Multiplies survival probability of a photon at given energy E with the logparabola spectrum.
        Returns: differential flux dN/dE = P(gamma->gamma) * logparabola
        '''

        dnde = self.log_parabola(en, norm, alpha, beta)
        return self.p[self.index] * dnde


    def read_probs(self):
        '''Reads in the probabilities of the pixel the code is working on.
        Returns: nothing, sets self.p to correct array.
        '''
        dat = np.loadtxt(prob_path)
        print(f'reading probabilities: {prob_path}')
        self.p = dat


    def propagation(self, EGeV, m=1., g=1., B0=10., seed=None, nsim=1):
        '''Runs the propagation in magnetic fields, copied from me-manu, NGC1257 example notebook.
        Returns: Probabilities of finding photon (transversal1, 2) and ALP.
        '''
	
        alp = ALP(m, g)
        ngc1275 = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
        pin = np.diag((1.,1.,0.)) * 0.5
        lambda_max = 35.
        lambda_min = 0.7
        kL = 2. * np.pi / lambda_max
        kH = 2. * np.pi / lambda_min
        m = ModuleList(alp, ngc1275, pin = pin, EGeV = EGeV)
        print('m:'), m.alp.m
        print('g:'), m.alp.g
        print('B:'), B0
        m.add_propagation("ICMGaussTurb",
                          0, # position of module counted from the source. 
                          nsim = nsim, # number of random B-field realizations
                          B0 = B0,  # rms of B field
                          n0 = 39.,  # normalization of electron density
                          n2 = 4.05, # second normalization of electron density, see Churazov et al. 2003, Eq. 4
                          r_abell = 500., # extension of the cluster
                          r_core = 80.,  # electron density parameter, see Churazov et al. 2003, Eq. 4
                          r_core2 = 280., # electron density parameter, see Churazov et al. 2003, Eq. 4
                          beta = 1.2,  # electron density parameter, see Churazov et al. 2003, Eq. 4
                          beta2= 0.58, # electron density parameter, see Churazov et al. 2003, Eq. 4
                          eta = 0.5, # scaling of B-field with electron denstiy
                          kL = kL, # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012 
                          kH = kH,  # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012
                          q = -2.80, # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012
                          seed=seed    # random seed for reproducability, set to None for random seed.
                          )
        m.add_propagation("EBL",1, model = 'dominguez') # EBL attenuation comes second, after beam has left cluster
        m.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS') # finally, the beam enters the Milky Way Field
        px,py,pa = m.run()
        self.px = px
        self.py = py
        self.pa = pa
        self.p = px + py
        # np.savetxt('/nfs/astrop/n1/kuhlmann/NGC_1275/sim/p_values_2pi.dat', self.p, fmt='%1.9e')
    

    def fit_routine(self):
        '''Does fitting routine for GTAnalysis instances.
        Returns: dictionary of self.gta.fit()
        '''
        # DONT USE THIS, IT'S WRONG! PLAIN WRONG!
        self.gta.free_sources(free=False)
        self.gta.free_source(source)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        self.gta.free_sources(distance=3, pars='norm')
        self.gta.free_sources(minmax_ts=[100, None], pars='norm')
        dloglike=1
        i=0
        while dloglike > 1e-5:
            print('optimizing...')
            o = self.gta.optimize()
            dloglike = o['dloglike'] / o['loglike0']
            i += 1
            if i >=10:
                print('10th iteration, will stop now')
                break
        fit_results = self.gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        if fit_results['fit_quality'] != 3:
            print('something fishy')
        return fit_results


    def cost_func_w_alps(self, norm, alpha, beta):
        print(norm, alpha, beta)
        dnde = self.survival_logparabola(self.en, norm, alpha, beta)
        # print 'dnde', dnde
        # print 'setting dnde'
        self.gta.set_source_dnde(self.name, dnde)
        like = self.gta.like()
        # print('like', like)
        return like


    def get_init_vals(self):
        '''Get values from logparabola-fitted model,
        multiplied by "scale" from model.xml file.
        '''

        self.norm = self.src_model['param_values'][0] * 1.0e11
        self.alpha = self.src_model['param_values'][1]
        self.beta = self.src_model['param_values'][2] * 1.0e1
        self.norm_err = self.src_model['param_errors'][0] * 1.0e11
        self.alpha_err = self.src_model['param_errors'][1]
        self.beta_err = self.src_model['param_errors'][2] * 1.0e1
        self.Eb = self.src_model['param_values'][3]
	# self.Eb = 884.

    def fit(self):
        '''Does the fitting stuff to get the new parameters of photon survival prob * logparabola.
        '''

        m = Minuit(self.cost_func_w_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta,\
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err, print_level=1)
        res = m.migrad()
        self.res.append(res)
        self.ll_minuit[self.index - start] = - gta.like()
        self.gta.free_sources(free=False)
        # self.gta.write_roi('minuit')
        self.gta.free_source(self.name)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        # self.gta.free_sources(distance=3, pars='norm')
        # self.gta.free_sources(minmax_ts=[100, None], pars='norm')
        
        o = gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        self.ll_fit[self.index - start] = - gta.like()
        """
        self.gta.load_roi('minuit')
        self.gta.free_source(self.name)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        self.gta.free_sources(distance=3, pars='norm')
        self.gta.free_sources(minmax_ts=[100, None], pars='norm')
        self.ll_fit_full[self.index] = - gta.like()
        """
    def write_outdata(self):
        for c, v in enumerate(self.res):
            try:
                self.outdata[c, 1] = self.ll_minuit[c]
            except:
                self.outdata[c, 1] = np.nan
            try:
                self.outdata[c, 0] = self.init_like
            except:
                self.outdata[c, 0] = np.nan
            try:
                self.outdata[c, 2] = self.ll_fit[c]
            except:
                self.outdata[c, 2] = np.nan
        # old_data = np.loadtxt(f'{loglike_path}/out_free_source_test_{which_gm:03}.dat')
        # self.outdata = np.append(old_data, self.outdata, axis=0)
        # np.savetxt(f'{loglike_path}/out_opt_{which_gm}.dat', self.outdata, \
        #            header='ll_final norm norm_err alpha alpha_err beta beta_err ll_init', fmt='%1.9e')
        # np.savetxt(f'{loglike_path}/out_{which_gm:03}_{which_range}.dat', self.outdata, \
        #            header='ll_init ll_minuit ll_fit', fmt='%1.9e') 
        try:
            save_data = np.loadtxt(f'{loglike_path}/out_{which_gm:03}.dat')
        except:
            save_data = np.zeros((100, 3))
        save_data[nsim*which_range:nsim*(which_range+1)] = self.outdata    
        np.savetxt(f'{loglike_path}/out_{which_gm:03}.dat', save_data, fmt='%1.9e')
'''Setup starts here.'''
print(f'is roi_file a valid path?', os.path.isfile(roi_file))

gta = GTAnalysis.create(roi_file, config=cwd+'/config_modified.yaml')

#actual simulation related stuff starts here
test = janitor(source, gta)
test.gta.free_sources(free=False)
test.gta.free_source(test.name)
#test.gta.free_source('galdiff')
test.src_model = test.gta.get_src_model(test.name)    # extract initial values for loglike fitting once
test.init_like = - test.gta.like()
test.gta.set_source_spectrum(test.name, spectrum_type='FileFunction')    # switch  to FF, get E
test.logEMeV, test.init_dnde = test.gta.get_source_dnde(test.name)
# np.savetxt('/nfs/astrop/n1/kuhlmann/NGC_1275/energy_bins.dat', test.logEMeV)
# test.EGeV = np.power(10., test.logEMeV - 3.)
test.en = np.power(10., test.logEMeV)
test.get_init_vals()
# test.propagation(test.EGeV, nsim=nsim, m=m, g=g, seed=seed)
test.read_probs()
start = which_range * nsim
end = (which_range + 1) * nsim
for i in range(start, end):
    test.index = i
    print('doing simulation number', str(i))
    test.fit()
    test.write_outdata()
    if i != end - 1:
        test.gta.load_roi(roi_file)
        test.gta.set_source_spectrum(test.name, spectrum_type='FileFunction')
    else:
        pass
# test.write_outdata()
# ts_fit = np.sort(2 * (test.outdata[:, 2] - test.outdata[:, 0]))
# ts_minuit = np.sort(2 * (test.outdata[:, 1] - test.outdata[:, 0]))
# test.ts_95 = np.array([ts_fit[8], ts_minuit[8]])
# np.savetxt(f'{ts_path}/ts_9_{which_gm}.dat', test.ts_95)
