#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:27:00 2020

@author: David
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ncx2
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
nbins = 30
np.random.seed(420)
fig = plt.figure(1)
ax = fig.add_subplot(111)
# parameters of non central chi squared distribution
df, nc = 10.09, 2.51
# plot the pdf
x = np.linspace(0, 43, num=1000)
y = ncx2.pdf(x, df, nc)
ax.plot(x, y, label=f'pdf, df={df}, nc={nc}')


# generate some random numbers, do some histogram things
random = ncx2.rvs(df, nc, size=1000)
n, bins, patches = ax.hist(random, bins=nbins, label='random numbers', density=True)

# do fit from random numbers
# get x places:
dx = bins[1] - bins[0]
x_r = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, num=len(n))
def fit_func(x, df, nc):
    return ncx2.pdf(x, df, nc)
popt, pcov = curve_fit(fit_func, x_r, n)
# and overplot the fitted curve
df_fit = popt[0]
nc_fit = popt[1]
#ax.plot(x, ncx2.pdf(x, df_fit, nc_fit), label=f'fit, $df={df_fit:1.2f}\pm {np.sqrt(pcov[0, 0]):1.2f}, nc={nc_fit:1.2f}\pm{np.sqrt(pcov[1, 1]):1.2f}$')
ax_cdf = ax.twinx()
ax.set_xlim((0, 40))
x_space = np.linspace(0, 40, num=41)
#x_labelspace = 
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
xlim = ax.get_xlim()
ax_cdf.hist(random, density=True, bins=nbins, cumulative=True, histtype='step')
#x_cdf.plot(x, ncx2.cdf(x, df, nc), c='red')
#ax_cdf.plot(x, ncx2.cdf(x, df_fit, nc_fit), c='yellow')
cl_95 = ncx2.ppf(0.954, df, nc)
print(cl_95)
cl_99 = ncx2.ppf(0.997, df, nc)
ax_cdf.plot((cl_95, cl_95), (0, 1), c='black')
ax_cdf.plot((cl_99, cl_99), (0, 1), c='black')
ax_cdf.set_ylim((0, 1))
ax.plot(())
ax.plot()
ax.legend()
