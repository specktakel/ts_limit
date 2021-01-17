import numpy as np
from gammaALPs import Source, ALP, ModuleList
import gc

class Probs():
    def __init__(self, log10MeV, g=1, m=1, nsim=1, B0=10., seed=None, ppb=10, lambda_min=0.7, lambda_max=35., q=-2.80):
        self.log10MeV = log10MeV
        self.B0 = B0
        self.nsim = nsim
        self.g = g
        self.m = m
        self.seed = seed
        self.ppb = ppb
        self.pin = np.diag((1.0, 1.0, 0.)) * 0.5
        self.q = q
        self.kL = 2. * np.pi / lambda_max
        self.kH = 2. * np.pi / lambda_min
        self.setup_energy()
        self.alp = ALP(self.m, self.g)
        self.source = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
        self.mod = self.module()


    def setup_energy(self):
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


    def propagation(self):
        p_gamma_x, p_gamma_y, p_gamma_a = self.mod.run()
        p_gamma_tot = p_gamma_x + p_gamma_y
        p_out = p_gamma_tot.reshape((self.nsim, self.nbins, self.ppb))
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


if __name__ == "__main":
    sys.exit()

