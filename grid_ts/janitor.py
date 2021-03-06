import gc
from iminuit import Minuit
import os
import yaml
import numpy as np
from fermipy.gtanalysis import GTAnalysis
import psutil
process = psutil.Process(os.getpid())

'''Class that handles the grid analysis, e.g. selection of g/m, fitting, data handling.
'''

class Janitor:
    '''Because janitors do the work.
    '''

    def __init__(self, which_gm, which_roi, which_range, paths, load_probs=False, use_weights=False, index=-2):
        self.which_gm = which_gm
        # which_roi is part of random seed generation,
        # need to pick some value in wrapper still
        self.which_roi = which_roi
        self.which_range = which_range
        self.use_weights = use_weights
        self.index = index
        self.path_dict = paths
        self.set_up_paths()
        self.write_yaml_config()
        self.init_arrays()
        self.find_gm(which_gm)
        self.set_up_roi()
        self.load_probs = load_probs
        if not self.load_probs:
            self.init_probs()
        else:
            self.read_probs()


    def init_probs(self):
        from ts_limit.grid_survival_prob.probs import Probs
        self.prob = Probs(self.logEMeV, self.g, self.m)


    def find_gm(self, which_gm):
        '''Finds from a 30 by 30 grid the correct value of g and m
        '''
        g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
        m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
        grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
        for i in range(g_space.shape[0]):
            for j in range(m_space.shape[0]):
                grid[i, j, :] = g_space[i], m_space[j]
        grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))
        self.g = grid[which_gm, 0]
        self.m = grid[which_gm, 1]


    def set_up_paths(self):
        self.save_path = self.path_dict['save_path']
        self.save_dir = self.path_dict['save_dir']
        self.prob_path = self.path_dict.get('prob_path')
        self.roi_file = self.path_dict['roi_file']
        self.roi_dir = self.path_dict['roi_dir']
        self.package_path = self.path_dict['package_path']
        self.config_path = self.path_dict['config_path'] 


    def set_up_roi(self):
        '''Loads fermipy GTAnalysis object, sets up sources,
        loads some dictionaries of values for later use.
        '''
        self.gta = GTAnalysis.create(self.roi_file, config=self.config_path)
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.src_model = self.gta.get_src_model(self.name)
        self.init_like = - self.gta.like()
        self.get_init_vals()
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')
        self.logEMeV, self.init_dnde = self.gta.get_source_dnde(self.name)
        self.EMeV = np.power(10., self.logEMeV)
        self.get_init_vals()


    def init_arrays(self): 
        self.outdata = np.zeros((1, 3), dtype=float)
        self.ll_minuit = np.zeros((1), dtype=float)
        self.ll_fit = np.zeros((1), dtype=float)
        self.reload_like = np.zeros((1), dtype=float)
        self.res = []
        dat = np.load(self.roi_file, allow_pickle=True).flat[0]
        self.nplike = dat['roi']['loglike']


    def write_yaml_config(self):
        roi_dir = self.roi_dir
        path_to_conda = os.environ['CONDA_PREFIX']
        cwd = os.getcwd()
        with open(f'{self.config_path}', 'r') as i:
            config = yaml.safe_load(i)

        config['fileio']['workdir'] =  f'{roi_dir}'
        config['fileio']['outdir'] = f'{roi_dir}'
        config['fileio']['logfile'] = f'{roi_dir}/fermipy.log'
        config['data']['ltcube'] = f'{roi_dir}/ltcube_00.fits'
        config['model']['galdiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
        config['model']['isodiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
        config['logging']['verbosity'] = 4
        self.name = config['selection']['target']
        with open(f'{roi_dir}/config_modified.yaml', 'w') as o:
            yaml.dump(config, o)
        self.config_path = f'{roi_dir}/config_modified.yaml'


    def log_parabola(self, x, norm, alpha, beta):
        '''Log parabola.
        Returns: differential flux dN/dE
        '''

        dnde = norm * np.power((x / self.Eb) , -(alpha + beta * np.log((x / self.Eb))))
        return dnde

    
    def survival_logparabola(self, en, norm, alpha, beta):
        '''Multiplies survival probability of a photon at given energy E with the logparabola spectrum.
        Returns: differential flux dN/dE = P(gamma->gamma) * logparabola
        Uses only the first entry of self.p, so for reading in probs this will always use the first set of probabilities!
        '''

        dnde = self.log_parabola(en, norm, alpha, beta)
        if self.load_probs:
            return self.p[self.index] * dnde
        else:
            return self.p[0] * dnde


    def read_probs(self):
        '''Reads in the probabilities of the pixel the code is working on.
        Returns: nothing, sets self.p to correct array.
        '''
        dat = np.loadtxt(self.prob_path)
        print(f'reading probabilities: {self.prob_path}')
        self.p = dat


    def gen_probs(self):
        '''Generates probabilities on the fly.
        '''
        seed = int(f'{self.which_roi}{self.which_gm:03}{self.index:02}')
        self.p = self.prob.fractured_propagation(seed, weights=self.use_weights, index=self.index)
        # np.savetxt(f'{self.save_dir}/p_{seed}_energy_weight.dat', self.p)


    def cost_func_w_alps(self, norm, alpha, beta):
        '''Cost function used for fitting of spectrum with ALPs.
        Returns: loglike (<0).
        '''
        print(norm, alpha, beta, self.Eb)
        dnde = self.survival_logparabola(self.EMeV, norm, alpha, beta)
        self.gta.set_source_dnde(self.name, dnde)
        like = self.gta.like()
        return like


    def get_init_vals(self):
        '''Get values from logparabola-fitted model,
        multiplied by "scale" from model.xml file.
        '''
        
        self.norm = self.src_model['param_values'][0]
        self.alpha = self.src_model['param_values'][1]
        self.beta = self.src_model['param_values'][2]
        self.norm_err = self.norm * 0.05
        self.alpha_err = self.alpha * 0.05
        self.beta_err = self.beta * 0.05
        self.Eb = self.src_model['param_values'][3] 
        print(self.norm, self.norm_err)
        print(self.alpha, self.alpha_err)
        print(self.beta, self.beta_err)
        print(self.Eb)
        _sources = self.gta.get_sources()
        self.sources = {}
        for src in _sources:
            self.sources[src.name] = src.params.copy()
        '''
        self.norm = self.sources[self.name]['norm']['value']
        self.alpha = self.sources[self.name]['alpha']['value']
        self.beta = self.sources[self.name]['beta']['value']
        self.norm_err = self.sources[self.name]['norm']['error']
        self.alpha_err = self.sources[self.name]['alpha']['error']
        self.beta_err = self.sources[self.name]['beta']['error']
        self.Eb = self.sources[self.name]['Eb']['value']
        '''

    def reload_gta(self):
        self.gta.load_roi(self.roi_file)
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')


    def bootleg_reload(self):
        self.gta.free_sources(free=False)
        self.gta.set_source_spectrum(self.name, spectrum_type='LogParabola', spectrum_pars=self.src_model['spectral_pars'])
        for k, v in self.sources.items():
            for param, param_dict in v.items():
                try:
                    self.gta.set_parameter(k, param, param_dict['value'], error=param_dict['error'])
                except:
                    pass
        print('reload like:', -self.gta.like())
        print('init like:', self.init_like)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')


    def fit(self):
        '''Does the fitting stuff to get the new parameters of photon survival prob * logparabola.
        '''
        if not self.load_probs:
            self.gen_probs()
        else:
            pass
        self.reload_like = - self.gta.like()
        m = Minuit(self.cost_func_w_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta,\
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err, print_level=1)
        # m.tol = 5
        res = m.migrad()
        self.res.append(res)
        print(self.gta.like())
        prelim_dnde = self.survival_logparabola(self.EMeV, m.values['norm'], m.values['alpha'], m.values['beta'])
        self.gta.set_source_dnde(self.name, prelim_dnde)
        print(self.gta.like())
        self.ll_minuit = - self.gta.like()
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        o = self.gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        self.ll_fit = - self.gta.like()


    def write_outdata(self):
        '''Write outdata to some specified filepath in a specified location. Differs between rerun and initial run.
        '''
        self.outdata[0, 0] = self.reload_like
        self.outdata[0, 1] = self.ll_minuit
        self.outdata[0, 2] = self.ll_fit
        try:
            save_data = np.loadtxt(self.save_path)
        except IOError:
            save_data = np.zeros((100, 3), dtype=float)
        save_data[self.index] = self.outdata
        header=f'init minuit fit, nplike: {self.nplike:5.4f}, initlike: {self.init_like:5.4f}'
        np.savetxt(self.save_path, save_data, fmt="%5.4f", header=header)


if __name__ == "__main__":
    sys.exit()

