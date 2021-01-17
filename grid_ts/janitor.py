import gc
from probs import Probs
from iminuit import Minuit


'''Class that handles the grid analysis, e.g. selection of g/m, fitting, data handling.
'''

class Janitor:
    '''Because janitors do the work.
    '''

    def __init__(self, which_gm, which_roi, which_range, paths, rerun=False, load_probs=True):
        self.rerun = rerun
        self.path_dict = paths
        self.set_up_roi()
        
        self.reload_like = np.zeros((nsim), dtype=float)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')
        
        # self.read_probs()
        self.load_probs=load_probs
        if not self.load_probs:
            self.prob = Probs(self.logEMeV, self.g, self.m)
        else:
            pass


    def find_gm(self):
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

    def set_up_roi(self)
        '''Loads fermipy GTAnalysis object, sets up sources,
        loads some dictionaries of values for later use.
        '''
        self.gta = GTAnalysis.create(self.roi_file, config=self.config_path)
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.src_model = self.gta.get_src_model(self.name)
        self.init_like = - self.gta.like()
        self.logEMeV, self.init_dnde = self.gta.get_source_dnde(self.name)
        self.EMeV = np.power(10., self.logEMeV)
        self.get_init_vals()

    def init_arrays(self): 
        self.outdata = np.zeros((nsim, 3), dtype=float)
        self.ll_minuit = np.zeros((nsim), dtype=float)
        self.ll_fit = np.zeros((nsim), dtype=float)
        self.res = []
        dat = np.load(roi_file, allow_pickle=True).flat[0]
        self.nplike = dat['roi']['loglike']

    def write_yaml_config(self):
        roi_dir = self.roi_dir
        path_to_conda = os.environ['CONDA_PREFIX']
        cwd = os.getcwd()
        with open(f'{cwd}/config_local.yaml', 'r') as i:
            config = yaml.safe_load(i)

        config['fileio']['workdir'] =  f'{roi_dir}'
        config['fileio']['outdir'] = f'{roi_dir}'
        config['fileio']['logfile'] = f'{roi_dir}/fermipy.log'
        config['data']['ltcube'] = f'{roi_dir}/ltcube_00.fits'
        config['model']['galdiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
        config['model']['isodiff'] = f'{path_to_conda}/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
        config['logging']['verbosity'] = 4
        self.name = config['selection']['target']
        with open(f'{cwd}/config_modified.yaml', 'w') as o:
            yaml.dump(config, o)
        self.config_path = 'config_modified.yaml'


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
        if self.load_probs:
            return self.p[self.index] * dnde
        else:
            return self.p[0] * dnde


    def read_probs(self):
        '''Reads in the probabilities of the pixel the code is working on.
        Returns: nothing, sets self.p to correct array.
        '''
        dat = np.loadtxt(prob_path)
        print(f'reading probabilities: {prob_path}')
        self.p = dat


    def gen_probs(self):
        '''Generates probabilities on the fly.
        '''
        if not hasattr(self.prob, 'mod'):
            self.prob.load_mod()
        else:
            pass    
        self.p = self.prob.propagation()
        self.prob.del_mod()
        unreachable = gc.collect()
        print("objects unreachable by gc:", unreachable)


    def cost_func_w_alps(self, norm, alpha, beta):
        '''Cost function used for fitting of spectrum with ALPs.
        Returns: loglike (<0).
        '''
        print(norm, alpha, beta)
        dnde = self.survival_logparabola(self.EMeV, norm, alpha, beta)
        self.gta.set_source_dnde(self.name, dnde)
        like = self.gta.like()
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


    def reload_gta(self):
        self.gta.load_roi(self.roi_file)
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')
            


    def fit(self):
        '''Does the fitting stuff to get the new parameters of photon survival prob * logparabola.
        '''
        # self.gen_probs()
        if self.rerun:
            self.temp = np.zeros((1, 3), dtype=float)
            self.temp[0, 0] = - self.gta.like()
        else:
            self.reload_like[self.index - start] = - self.gta.like()
        m = Minuit(self.cost_func_w_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta,\
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err, print_level=1)
        # m.tol = 5
        res = m.migrad()
        self.res.append(res)
        if self.rerun:
            self.temp[0, 1] = - self.gta.like()
        else:
            self.ll_minuit[self.index - start] = - self.gta.like()
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')
        o = gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        if rerun:
            self.temp[0, 2] = - self.gta.like()

        else:
            self.ll_fit[self.index - start] = - self.gta.like()


    def write_outdata(self):
        '''Write outdata to some specified filepath in a specified location. Differs between rerun and initial run.
        '''
        if self.rerun:
            save_data = np.loadtxt(f'{loglike_dir}/out_{which_gm:03}.dat')
            save_data[self.index] = self.temp
            np.savetxt(f'{loglike_dir}/out_{which_gm:03}.dat', save_data, fmt="%5.4f")
            del self.temp
        else:
            for c, v in enumerate(self.res):
                try:
                    self.outdata[c, 1] = self.ll_minuit[c]
                except:
                    self.outdata[c, 1] = 0
                try:
                    self.outdata[c, 0] = self.reload_like[c]
                except:
                    self.outdata[c, 0] = 0
                try:
                    self.outdata[c, 2] = self.ll_fit[c]
                except:
                    self.outdata[c, 2] = 0
            try:
                save_data = np.loadtxt(f'{loglike_dir}/out_{which_gm:03}.dat')
            except:
                save_data = np.zeros((100, 3))
            save_data[nsim*which_range:nsim*(which_range+1)] = self.outdata    
            np.savetxt(f'{loglike_dir}/out_{which_gm:03}.dat', save_data, fmt='%5.4f', \
                       header=f'init minuit fit, nplike: {self.nplike:5.4f}, loaded like: {self.init_like:5.4f}')



if __name__ == "__main__":
    sys.exit()

