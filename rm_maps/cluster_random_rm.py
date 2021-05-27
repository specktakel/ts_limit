from b_field import structured_field as s
import numpy as np
import numpy.ma as ma
from scipy.integrate import simps
from scipy.stats import norm
from matplotlib import cm
from gammaALPs.nel.icm import NelICM
from gammaALPs.bfields.gauss import Bgaussian
from gammaALPs.core import ModuleList, ALP, Source
from scipy.spatial.transform import Rotation as R_
import time


nel = NelICM(n0 = 39., n2 = 4.05, r_abell = 500., r_core = 80., r_core2 = 280., beta = 1.2, beta2= 0.58, eta = 0.5)

def init_gauss(B0, seed):
    alp = ALP(1, 1)
    source = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
    m = ModuleList(alp, source, pin=0.5 * np.diag((1, 1, 0)), EGeV=np.linspace(2, 2.6, num=10))
    m.add_propagation("ICMGaussTurb", 
                      0,
                      nsim = 1,
                      B0 = B0,
                      n0 = 39.,
                      n2 = 4.05,
                      r_abell = 500.,
                      r_core = 80.,
                      r_core2 = 280.,
                      beta = 1.2,
                      beta2= 0.58,
                      eta = 0.5,
                      kL = 2 * np.pi / 35, 
                      kH = 2 * np.pi / 0.7,
                      q = -2.80,
                      seed = seed
                      )
    #m.add_propagation("EBL",1, model = 'dominguez')
    #m.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS')
    return m.modules[0]._Bfield_model

def generate_B_random(Bg, num, z):
    B, psi = Bg.new_Bn(z, nsim=num)
    B_ = np.zeros((num, z.size))
    for c_, v_ in enumerate(B):
        B_[c_] = v_ * np.sin(psi[c_])
    return B_



def make_random_rm(x, y, z, B):
    # x, y are single coordinates, z is vector along which B is calculated
    r = np.sqrt(x**2 + y**2 + z**2)
    #scale B field, fixed eta to 0.5
    B_scaled = B * np.power(nel(r) / nel(0), 0.5)
    
    rm = 812 * simps(B_scaled * np.where(r < 500, nel(r) * 1e-3, 0), z)
    return rm

def make_rm_dist(x, y, B0, rm_num, z_steps, seed):
    Bgaus = init_gauss(B0, seed)
    R = 500
    dL = R / z_steps
    z = np.linspace(dL / 2, R - dL / 2, num=z_steps, endpoint=True)
    Bfield = generate_B_random(Bgaus, rm_num, z)
    rm = np.zeros(rm_num)
    for c_, B in enumerate(Bfield):
        rm[c_] = make_random_rm(x, y, z, B)
    print(rm)
    pars = norm.fit(rm[1:])
    return pars[1]

x = np.linspace(0, 500, num=50)
sigma = np.zeros(x.shape)
start = time.time()
for c_, x_ in enumerate(x):
    sigma[c_] = make_rm_dist(x_, 0, 10, 100, 5000, None)
    break
end = time.time()
print('elapsed time', end - start)
