#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:11:48 2021

@author: julian
"""
from b_field import structured_field
from gammaALPs import Source, ALP, ModuleList
from ts_limit.grid import Grid
import numpy as np
b = structured_field(8.3, 93, 225)

log10MeV = np.loadtxt('../grid_survival_prob/energy_bins.dat')
EGeV = np.power(10, log10MeV - 3.)
grid = Grid()
gm = grid.get_gm(645)
#g = gm[0]
#m = gm[1]
g = 1e-1
m = 2
n_el = 4e-3
alp = ALP(m,g)
source = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
pin = 0.5 * np.diag((1., 1., 0.))
ml = ModuleList(alp, source, pin=pin, EGeV=EGeV)
ml.add_propagation("Array", 0, Btrans=b.b_trans, psi=b.angle, nel=n_el, dL=b.dL_vec)
px, py, pa = ml.run()
plt.plot(EGeV, px+py)
plt.xscale('log')
