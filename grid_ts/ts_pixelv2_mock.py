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


try:
    num = int(sys.argv[4])
    rerun = True
    print("rerun some args...")
except IndexError:
    rerun = False
    print("initial run")


'''Set up roi/parameters space'''
nsim = 25
grid = np.linspace(0, 900, num=900, endpoint=False, dtype=int)
# roi_path = f'{cwd}/roi_{which_roi}'
# print('roi_path:', roi_path, os.path.isdir(roi_path))


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

# roi_path = f"{cwd}/fits"      # for testing on astro-wgs1[1, 2]
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
prob_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/new_probs/roi_{which_roi}/prob_{which_gm:03}.dat'
# loglike_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/roi_{which_roi}'
# ts_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/orig_data/updated_lite_ts'
loglike_path = f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/roi_{which_roi}'
roi_file = f'{roi_path}/roi_{which_roi}.npy'
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
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.src_model = self.gta.get_src_model(self.name)
        self.init_like = - self.gta.like()
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')
        self.logEMeV, self.init_dnde = self.gta.get_source_dnde(self.name)
        self.en = np.power(10., self.logEMeV)
        self.get_init_vals()
        self.read_probs()


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
        if rerun:
            self.temp = np.zeros((1, 3), dtype=float)
            self.temp[0, 0] = self.init_like
        else:
            pass
        m = Minuit(self.cost_func_w_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta,\
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err, print_level=1)
        res = m.migrad()
        self.res.append(res)
        if rerun:
            self.temp[0, 1] = - gta.like()
        else:
            self.ll_minuit[self.index - start] = - gta.like()
        self.gta.free_sources(free=False)
        # self.gta.write_roi('minuit')
        self.gta.free_source(self.name)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        # self.gta.free_sources(distance=3, pars='norm')
        # self.gta.free_sources(minmax_ts=[100, None], pars='norm')
        
        o = gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        if rerun:
            self.temp[0, 2] = - gta.like()

        else:
            self.ll_fit[self.index - start] = - gta.like()


    def write_outdata(self):
        if rerun:
            save_data = np.loadtxt(f'{loglike_path}/out_{which_gm:03}.dat')
            save_data[self.index] = self.temp
            np.savetxt(f'{loglike_path}/out_{which_gm:03}.dat', save_data, fmt="%1.9e")
            del self.temp
        else:
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
            try:
                save_data = np.loadtxt(f'{loglike_path}/out_{which_gm:03}.dat')
            except:
                save_data = np.zeros((100, 3))
            save_data[nsim*which_range:nsim*(which_range+1)] = self.outdata    
            np.savetxt(f'{loglike_path}/out_{which_gm:03}.dat', save_data, fmt='%1.9e')
       


'''Setup starts here.'''
print(f'is roi_file a valid path?', os.path.isfile(roi_file))


if rerun:
    try:
        data = np.loadtxt(f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/roi_{which_roi}/out_{which_gm:03}.dat")
        indices = []
        if data.shape[0] == 100 and data.shape[1] == 3:
            for c, line in enumerate(data):
                if np.any(np.isclose(line, np.zeros(line.shape))):
                    indices.append(c)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("file has wrong shape or is not found")
        data = np.zeros((100, 3), dtype=float)
        np.savetxt(f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/outdata/roi_{which_roi}/out_{which_gm:03}.dat", data)
        start = which_range * nsim
        end = (which_range + 1) * nsim
        indices = [i for i in range(start, end)]
    

else:
    start = which_range * nsim
    end = (which_range + 1) * nsim    
    indices = [i for i in range(start, end)]

print(indices)
# sys.exit()
gta = GTAnalysis.create(roi_file, config=cwd+'/config_modified.yaml')
test = janitor(source, gta)
test.indices = indices

for i in indices:
    test.index = i
    print('doing simulation number', str(i))
    test.fit()
    test.write_outdata()
    if i != indices[-1]:
        test.gta.load_roi(roi_file)
        test.gta.set_source_spectrum(test.name, spectrum_type='FileFunction')
    else:
        pass
# test.write_outdata()
# ts_fit = np.sort(2 * (test.outdata[:, 2] - test.outdata[:, 0]))
# ts_minuit = np.sort(2 * (test.outdata[:, 1] - test.outdata[:, 0]))
# test.ts_95 = np.array([ts_fit[8], ts_minuit[8]])
# np.savetxt(f'{ts_path}/ts_9_{which_gm}.dat', test.ts_95)
