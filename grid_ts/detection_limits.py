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
import matplotlib.gridspec as gridspec
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
nbins = 30
np.random.seed(420)
fig = plt.figure(1, dpi=150, figsize=(10, 12))
axs = []
gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
ax = fig.add_subplot(gs[0, :])
axs.append(fig.add_subplot(gs[1, 0]))
axs.append(fig.add_subplot(gs[1, 1]))

ax.set_xlim((0, 42))
ax.set_title('Analysis of PE')
ax.set_xlabel('TS')
ax.set_ylabel('pdf, normalized')
data = np.loadtxt('ts_95_max_values.dat')
ts_max = data[:, 0].copy()
ts_95 = data[:, 1].copy()
# parameters of non central chi squared distribution
#df, nc = 10.09, 2.51
# plot the pdf
x = np.linspace(0, 43, num=1000)
#y = ncx2.pdf(x, df, nc)
#ax.plot(x, y, label=f'pdf, $df={df}, nc={nc}$', color='black', linewidth=0.8)


# generate some random numbers, do some histogram things
size = 1000
#random = ncx2.rvs(df, nc, size=size)
n, bins, patches = ax.hist(ts_max, bins=nbins, label='max TS per PE', density=True, color='grey')
cum_bins = bins.copy()
cum_bins[-1] = 50
# do fit from random numbers
# get x places:
dx = (bins[1] - bins[0]) / 2
x_r = np.linspace(bins[0] + dx / 2, bins[-1] - dx / 2, num=len(bins))
#ax.plot(bins+dx, np.zeros(bins.shape[0]), 'o')


def fit_func(x, df, nc):
    return ncx2.pdf(x, df, nc)



popt, pcov = curve_fit(fit_func, bins[:-1]+dx, n, p0=(20, 10))
# and overplot the fitted curve
df_fit = popt[0]
nc_fit = popt[1]
df_err = np.sqrt(pcov[0, 0])
nc_err = np.sqrt(pcov[1, 1])
ax.plot(x, ncx2.pdf(x, df_fit, nc_fit),
        label=F'fit, $df={df_fit:1.2f} \pm {df_err:1.2f}, nc={nc_fit:1.2f} \pm {nc_err:1.2f}$', linewidth=0.8)
ax_cdf = ax.twinx()
ax_cdf.set_ylabel('cdf')
ax.set_xlim((0, 40))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
xlim = ax.get_xlim()
ax_cdf.hist(ts_max, density=True, bins=cum_bins, cumulative=True,
            histtype='step', color='black', linewidth=0.6)
#ax_cdf.plot(x, ncx2.cdf(x, df, nc), c='red', linewidth=0.8)
#cl_95 = ncx2.ppf(0.954, df, nc)
#cl_99 = ncx2.ppf(0.997, df, nc)
#ax_cdf.plot((cl_95, cl_95), (0, 1), c='black', linestyle='-.', linewidth=0.5)
#ax_cdf.plot((cl_99, cl_99), (0, 1), c='black', linestyle='--', linewidth=0.5)
ax_cdf.set_ylim((0, 1))
#ax_cdf.text(cl_95, 0.5, '$2\sigma$', rotation=90)
#ax_cdf.text(cl_99, 0.5, '$3\sigma$', rotation=90)
ax_cdf.set_ylim((0, 1))


data_rv = rv_histogram((n, bins))


# cl from actual data:
cl_95_dat = data_rv.ppf(0.954)
cl_99_dat = data_rv.ppf(0.997)
exclusion_limits = np.array([cl_95_dat, cl_99_dat])
ax_cdf.plot((cl_95_dat, cl_95_dat), (0, 1), c='black', linestyle='-.',
            linewidth=0.5)
ax.text(cl_95_dat, 0.04, '$2\sigma$', rotation=90)
ax_cdf.plot((cl_99_dat, cl_99_dat), (0, 1), c='black', linestyle='--',
            linewidth=0.5, label='from data')
ax.text(cl_99_dat, 0.04, '$3\sigma$', rotation=90)

ax.legend(loc='upper right')
#ax.legend(loc='upper left')

#%%

# do the boostrapping stuff here
# idea: resample size=1000 samples from initial distribution, calculate the
# 2/3 sigma TS, respectively. Distribution of these: find +/- 34% boundaries,
# so in total the 68% CL limit 

n_resample = 10000
cl_sigma = np.zeros((2, n_resample))    # 2sigma, 3sigma
for i in range(n_resample):
    resample = data_rv.rvs(size=size)
    binned = np.histogram(resample, bins=bins)
    resampled_rv = rv_histogram(binned)
    cl_sigma[0, i] = resampled_rv.ppf(0.954)
    cl_sigma[1, i] = resampled_rv.ppf(0.997)

axs_cdf = np.array([ax_.twinx() for ax_ in axs])
#%%
sigma_pdf = []
bounds = np.zeros((2, 2))
for i in range(cl_sigma.shape[0]):
    n_1, bins_1, patches_1 = axs[i].hist(cl_sigma[i], density=True, bins=30)
    cdf_bins = bins_1.copy()
    cdf_bins[-1] +=10
    n_2, bins_2, patches_2 = axs_cdf[i].hist(cl_sigma[i], bins=cdf_bins, density=True,
                                             cumulative=True, histtype='step', color='red')
    sigma_pdf.append(rv_histogram((n_1, bins_1)))
    upper_bound = sigma_pdf[i].ppf(0.34 + sigma_pdf[i].cdf(exclusion_limits[i]))
    lower_bound = sigma_pdf[i].ppf(sigma_pdf[i].cdf(exclusion_limits[i]) - 0.34)
    bounds[i, :] = np.array([lower_bound, upper_bound])
    axs[i].set_xlabel('TS')
    axs[i].set_ylabel('pdf')
    axs_cdf[i].set_ylabel('cdf')
    axs_cdf[i].set_ylim((0, 1))
    axs_cdf[i].plot((exclusion_limits[i], exclusion_limits[i]), (0, 1), c='black', linestyle='-.',
            linewidth=0.5)
    axs[i].text(exclusion_limits[i], 0.04, '$2\sigma$', rotation=90)
    axs[i].set_xlim((bins_1[0], bins_1[-1]))
    
plt.show()




