from iminuit import Minuit
import numpy as np
import yaml
'''Class that handles mostly fitting during grad analysis.
Moved most other stuff outside again to the wrapper.
It was getting really messy with all the path handling
required for simulated and real data.
And then the other magnetic field...'''

class Janitor:
    '''Because janitors do the work.
    '''
    def __init__(self, gta, bracket=None, **kwargs):
        if bracket:
            self.ae_mod_log = np.log10(np.loadtxt(bracket))
            print(self.ae_mod_log)
        self.gta = gta
        self.name = self.gta.config['selection']['target']
        self.kwargs = kwargs
        self.set_up_roi()
        self.get_init_vals()
        # self.outdata = np.zeros((1, 3))
        self.outdata = np.zeros((1, 2))
        self.p = np.ones(self.logEMeV.shape)
        self.res = []
        self.reload_like = - self.gta.like()
        for c_, E in enumerate(self.EMeV):
            if E > 2e3:
                self.cut = c_
                break
        self.EMeV_below = self.EMeV[:c_]
        self.EMeV_above = self.EMeV[c_:]
        self.param_vec = []

    def set_up_roi(self):
        '''Loads fermipy GTAnalysis object, sets up sources,
        loads some dictionaries of values for later use.
        '''
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.src_model = self.gta.get_src_model(self.name)
        self.init_like = - self.gta.like()
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')
        self.logEMeV, self.init_dnde = self.gta.get_source_dnde(self.name)
        self.EMeV = np.power(10., self.logEMeV)

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
        return self.p * dnde

    def interpolate_ae(self):
        print('interpolating')
        log_interp = np.interp(self.logEMeV, self.ae_mod_log[:, 0], self.ae_mod_log[:, 1])
        return np.power(10, log_interp)

    def cost_func_w_alps(self, norm, alpha, beta):
        '''Cost function used for fitting of spectrum with ALPs.
        Returns: loglike (<0).
        '''
        # print(norm, alpha, beta, self.Eb)
        dnde = self.survival_logparabola(self.EMeV, norm, alpha, beta)
        self.gta.set_source_dnde(self.name, dnde)
        like = self.gta.like()
        return like

    def cost_func_w_broken_sys(self, norm, alpha, beta, below2, above2):
        # print(norm, alpha, beta, self.Eb, below2, above2)
        dnde = self.logparabola(self.EMeV, norm, alpha, beta)
        # cut at 2GeV, below and above treated with different fit parameter
        scale_below = self.sys_err(below2, self.EMeV_below)
        scale_above = self.sys_err(above2, self.EMeV_above)
        scale = np.hstack((scale_below, scale_above))
        self.gta.set_source_dnde(self.name, dnde * scale)
        like = self.gta.like()
        return like

    def cost_func_w_broken_sys_alps(self, norm, alpha, beta, below2, above2):
        # print(norm, alpha, beta, self.Eb, below2, above2)
        dnde = self.survival_logparabola(self.EMeV, norm, alpha, beta)
        # cut at 2GeV, below and above treated with different fit parameter
        scale_below = self.sys_err(below2, self.EMeV_below)
        scale_above = self.sys_err(above2, self.EMeV_above)
        scale = np.hstack((scale_below, scale_above))
        self.gta.set_source_dnde(self.name, dnde * scale)
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

    def reload_gta(self):
        self.gta.load_roi(self.roi_file)
        self.gta.free_sources(free=False)
        self.gta.free_source(self.name)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')

    def bootleg_reload(self):
        self.gta.free_sources(free=False)
        self.set_logparabola()
        for k, v in self.sources.items():
            for param, param_dict in v.items():
                try:
                    self.gta.set_parameter(k, param, param_dict['value'], error=param_dict['error'])
                except:
                    pass
        print('reload like:', -self.gta.like())
        print('init like:', self.init_like)
        self.gta.set_source_spectrum(self.name, spectrum_type='FileFunction')

    def set_logparabola(self):
        self.gta.set_source_spectrum(self.name, spectrum_type='LogParabola', spectrum_pars=self.src_model['spectral_pars'])

    def fit(self):
        '''Does the fitting stuff to get the new parameters of photon survival prob * logparabola.
        '''
        self.reload_like = - self.gta.like()
        m = Minuit(self.cost_func_w_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta,
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err,
                   print_level=1)
        res = m.migrad()
        self.res.append(res)
        prelim_dnde = self.survival_logparabola(self.EMeV, m.values['norm'], m.values['alpha'], m.values['beta'])
        self.gta.set_source_dnde(self.name, prelim_dnde)
        self.ll_minuit = - self.gta.like()
        self.param_vec.append(np.array([m.values['norm'], m.values['alpha'], m.values['beta']]))
        # self.gta.free_sources(free=False)
        # self.gta.free_source(self.name)
        # self.gta.free_source('galdiff')
        # self.gta.free_source('isodiff')
        # o = self.gta.fit(optimizer='NEWMINUIT', reoptimize=True)
        # self.ll_fit = - self.gta.like()

    def fit_systematics(self):
        '''Does the fitting stuff to get the new parameters of photon survival prob * logparabola.
        '''
        m = Minuit(self.cost_func_w_broken_sys_alps, errordef=0.5, norm=self.norm, alpha=self.alpha, beta=self.beta, below2=0, above2=0,
                   error_norm=self.norm_err, error_alpha=self.alpha_err, error_beta=self.beta_err, error_below2=0.02, error_above2=0.02, 
                   limit_above2=(-1, 1), limit_below2=(-1, 1), print_level=1)
        res = m.migrad()
        if not res.fmin.is_valid:
            return False
        self.res.append(res)
        prelim_dnde = self.survival_logparabola(self.EMeV, m.values['norm'], m.values['alpha'], m.values['beta'])
        scale_below = self.sys_err(m.values['below2'], self.EMeV_below)
        scale_above = self.sys_err(m.values['above2'], self.EMeV_above)
        scale = np.hstack((scale_below, scale_above))
        self.gta.set_source_dnde(self.name, prelim_dnde * scale)
        self.ll_minuit = - self.gta.like()
        self.param_vec.append(np.array([m.values['norm'], m.values['alpha'], m.values['beta'], m.values['below2'], m.values['above2']]))
        return True

    def prepare_outdata(self):
        '''Write outdata to some specified filepath in a specified location. Differs between rerun and initial run.
        '''
        # self.outdata[0, 0] = self.reload_like
        self.outdata[0, 0] = self.init_like
        self.outdata[0, 1] = self.ll_minuit
        # self.outdata[0, 2] = self.ll_fit
        # self.header=f'init minuit fit, nplike: {self.nplike:5.4f}, initlike: {self.init_like:5.4f}'

    @classmethod
    def write_yaml(cls, inpath, outpath, **kwargs):
        with open(inpath, 'r') as i:
            config = yaml.safe_load(i)
        config['data']['ltcube'] = kwargs['data']['ltcube']
        config['fileio'] = kwargs['fileio']
        with open(outpath, 'w') as o:
            yaml.dump(config, o)
        print(config)

    def sys_err(self, x, E):
        '''Return interpolated between upper and lower limit, depending on energy as parameter.
        x is fit parameter used for linear interpolation.'''
        return 1 + x * (self.upper_bracket(E) - self.lower_bracket(E)) / 2
   
    @staticmethod
    def upper_bracket(e):
        try:
            out = np.zeros(e.size)
        except (TypeError, AttributeError):
            out = np.zeros(1)
            e = np.array([e])
        for c, i in enumerate(e):
            if i <= 100:
                out[c] = 1.03 + 0.14 * (2.0 - np.log10(i)) 
            elif i > 100 and i <= 100000:
                out[c] =  1.03
            elif i > 100000:
                out[c] = 1.03 + 0.12 * (np.log10(i) - 5.0)
        return out

    @staticmethod
    def lower_bracket(e):
        try:
            out = np.zeros(e.size)
        except (TypeError, AttributeError):
            out = np.zeros(1)
            e = np.array([e])
        for c, i in enumerate(e):
            if i <= 100:
                out[c] = 0.97 - 0.14 * (2.0 - np.log10(i)) 
            elif i > 100 and i <= 100000:
                out[c] =  0.97
            elif i > 100000:
                out[c] = 0.97 - 0.12 * (np.log10(i) - 5.0)
        return out


if __name__ == "__main__":
    sys.exit()

