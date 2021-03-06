import numpy as np
from gammaALPs import Source, ALP, ModuleList
import gc
from math import floor, ceil
import psutil
import os
from ts_limit.exceptions import UnphysicalError
process = psutil.Process(os.getpid())

class Probs():
    '''Initialises needed variables, ALPs object and source object.
       Energy for photon/ALP propagation needs to be set up manually, because we use
       oversampling of energy which changes the energy range multiple times.
       The same applies to the Module object.
       Returns: nothing
    '''

    def __init__(self, log10MeV, g=1, m=1, nsim=1, B0=10., seed=None, ppb=10, lambda_min=0.7, lambda_max=35., q=-2.80):
        self.log10MeV = log10MeV
        self.B0 = B0
        self.nsim = nsim
        self.g = g
        self.m = m
        self.seed = seed
        self._seed = seed
        self.ppb = ppb
        self.pin = np.diag((1.0, 1.0, 0.)) * 0.5
        self.q = q
        self.kL = 2. * np.pi / lambda_max
        self.kH = 2. * np.pi / lambda_min
        # self.set_up_energy()
        self.alp = ALP(self.m, self.g)
        self.source = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
        # self.mod = self.module()


    def set_up_energy(self):
        self.nbins = self.log10MeV.shape[0]
        delta_e = self.log10MeV[1] - self.log10MeV[0]
        delta_p = delta_e / self.ppb
        log_e_space = np.linspace(self.log10MeV[0], self.log10MeV[-1] + delta_e, num=self.nbins * self.ppb, endpoint=False) - (self.ppb - 1) / 2 * delta_p
        self.EGeV = np.power(10, log_e_space - 3.)


    def module(self):
        m = ModuleList(self.alp, self.source, pin=self.pin, EGeV=self.EGeV)
        m.add_propagation("ICMGaussTurb", 
                          0,
                          nsim = self.nsim,
                          B0 = self.B0,
                          n0 = 39.,
                          n2 = 4.05,
                          r_abell = 500.,
                          r_core = 80.,
                          r_core2 = 280.,
                          beta = 1.2,
                          beta2= 0.58,
                          eta = 0.5,
                          kL = self.kL, 
                          kH = self.kH,
                          q = self.q,
                          seed = self.seed
                          )
        m.add_propagation("EBL",1, model = 'dominguez')
        m.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS')
        return m


    def propagation(self, seed=None, weights=None, index=-2):
        '''Experimental feature: binning with spectral index thingy as weight
        Need to get the ppb-logspace from setup_energy method down here to be used as weights.
        Argument 'seed' overwrites the attribute seed!'''

        if seed is not None:
            self.seed = seed
        else:
            pass
        print("Seed:", self.seed)
        self.set_up_energy()
        self.load_mod()
        p_gamma_x, p_gamma_y, p_gamma_a = self.mod.run()
        p_gamma_tot = p_gamma_x + p_gamma_y
        p_out = p_gamma_tot.reshape((self.nsim, self.nbins, self.ppb))
        if weights is not None:
            weights = np.zeros((self.nsim, self.EGeV.shape[0]))
            weights[:] = np.power(self.EGeV, index)
            weights = weights.reshape((self.nsim, self.nbins, self.ppb))
            p_out = np.average(p_out, weights=weights, axis=-1)
        else:
            p_out = np.average(p_out, axis=-1)       
        return p_out


    def del_mod(self):
        del self.mod
        gc.collect()


    def load_mod(self):
        self.mod = self.module()


    def reload_module(self):
        self.del_mod()
        self.load_mod()


    def fractured_propagation(self, seed=None, weights=None, index=-2):
        try:
            self.del_mod()
        except AttributeError:
            pass
        print('seed:', seed)
        if seed is None:
            raise UnphysicalError
        log10MeV = self.log10MeV.copy()
        length = log10MeV.shape[0]
        parts = ceil(length / 20)
        p = np.zeros((self.nsim, length))
        for i in range(parts):
            print(i)
            self.log10MeV = log10MeV[i*20:(i+1)*20]
            self.set_up_energy()
            self.load_mod()
            # print("memory usage with modul:", process.memory_info().rss * 1e-6)
            temp = self.propagation(seed, weights, index)
            # print("memory usage after propagation:", process.memory_info().rss * 1e-6)
            self.del_mod()
            # print("memory usage after del:", process.memory_info().rss * 1e-6)
            p[:, i*20:(i+1)*20] = temp
        self.log10MeV = log10MeV.copy()
        # print(self.log10MeV)
        return p


if __name__ == "__main__":
    sys.exit()

