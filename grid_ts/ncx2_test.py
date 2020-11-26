#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:27:00 2020

@author: David
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ncx2
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
nbins = 30
np.random.seed(420)
fig = plt.figure(1, dpi=150)
ax = fig.add_subplot(111)
ax.set_xlim((0, 42))
ax.set_title('Analysis of PE')
ax.set_xlabel('TS')
ax.set_ylabel('pdf, normalized')

# parameters of non central chi squared distribution
df, nc = 10.09, 2.51
# plot the pdf
x = np.linspace(0, 43, num=1000)
y = ncx2.pdf(x, df, nc)
ax.plot(x, y, label=f'pdf, $df={df}, nc={nc}$', color='black', linewidth=0.8)


# generate some random numbers, do some histogram things
random = ncx2.rvs(df, nc, size=1000)
n, bins, patches = ax.hist(random, bins=nbins, label='random numbers', density=True, color='grey')
cum_bins = bins.copy()
cum_bins[-1] = 50
# do fit from random numbers
# get x places:
dx = (bins[1] - bins[0]) / 2
x_r = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, num=len(bins))
#ax.plot(bins+dx, np.zeros(bins.shape[0]), 'o')


def fit_func(x, df, nc):
    return ncx2.pdf(x, df, nc)



popt, pcov = curve_fit(fit_func, bins[:-1]+dx, n, p0=(20, 1))
# and overplot the fitted curve
df_fit = popt[0]
nc_fit = popt[1]
df_err = np.sqrt(pcov[0, 0])
nc_err = np.sqrt(pcov[1, 1])
ax.plot(x, ncx2.pdf(x, df_fit, nc_fit), label=F'fit, $df={df_fit:1.2f} \pm {df_err:1.2f}, nc={nc_fit:1.2f} \pm {nc_err:1.2f}$', linewidth=0.8)
ax_cdf = ax.twinx()
ax_cdf.set_ylabel('cdf')
ax.set_xlim((0, 40))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
xlim = ax.get_xlim()
ax_cdf.hist(random, density=True, bins=cum_bins, cumulative=True, histtype='step', color='black', linewidth=0.6)
ax_cdf.plot(x, ncx2.cdf(x, df, nc), c='red', linewidth=0.8)
cl_95 = ncx2.ppf(0.954, df, nc)
cl_99 = ncx2.ppf(0.997, df, nc)
ax_cdf.plot((cl_95, cl_95), (0, 1), c='black', linestyle='-.', linewidth=0.5)
ax_cdf.plot((cl_99, cl_99), (0, 1), c='black', linestyle='--', linewidth=0.5)
ax_cdf.set_ylim((0, 1))
ax_cdf.text(cl_95, 0.5, '$2\sigma$', rotation=90)
ax_cdf.text(cl_99, 0.5, '$3\sigma$', rotation=90)



# convert the actual data/random numbers into a pdf and cfd!
# n, bins is the data and bin boundaries, respectively
# works, but does not look so good! need improvement on optics
data_rv = rv_histogram((n, bins))
fig_2 = plt.figure(2, dpi=150)
ax_2 = fig_2.add_subplot(111)
ax_2.xaxis.set_major_locator(MultipleLocator(5))
ax_2.xaxis.set_minor_locator(MultipleLocator(1))
ax_2.set_xlim((0, 43))
y_rv = data_rv.pdf(bins)
ax_2.bar(bins, y_rv)

# cl from actual data:
cl_95_dat = data_rv.ppf(0.954)
cl_99_dat = data_rv.ppf(0.997)
ax_cdf.plot((cl_95_dat, cl_95_dat), (0, 1), c='yellow', linestyle='-.', linewidth=0.5)
ax_cdf.text(cl_95_dat, 0.4, '$2\sigma$', rotation=90)
ax_cdf.plot((cl_99_dat, cl_99_dat), (0, 1), c='yellow', linestyle='--', linewidth=0.5, label='from data')
ax_cdf.text(cl_99_dat, 0.4, '$3\sigma$', rotation=90)

ax_cdf.legend(loc='upper right')
ax.legend(loc='upper left')
