import numpy as np
from gammaALPs import Source, ALP, ModuleList
import os
import time
import sys
cwd = os.getcwd()
param_list = []

'''Script to pre-compute photon survival probabilities.
For nsim=100, ppb=4 and one parameter set of g, m, execution takes roughly 850 seconds,
meaning that this has to be distributed across the cluster in 900 jobs, as well.
'''


'''parameters for generation'''
param_list.append(int(sys.argv[1]))     # cheeky way to not delete for loop down there. Take $(ProcId) as arg for g/m combination.
# part = int(sys.argv[2])
seed = None
nsim = 50    # small for testing purposes
B0 = 10.
ppb = 10    # points per bin
header = f'seed: {seed}, nsim: {nsim}, B0: {B0}, ppb: {ppb}'
config = np.array([seed, nsim, B0, ppb])
# np.savetxt('/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/config.dat', config, header=header)
which_roi = int(sys.argv[2])

'''Set up parameter list'''

g_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in 1e-11 1/GeV
m_space = np.logspace(-1., 2., num=30, base=10.0, endpoint=True)    # in neV
grid = np.zeros((g_space.shape[0], m_space.shape[0], 2))
for i in range(g_space.shape[0]):
    for j in range(m_space.shape[0]):
        grid[i, j, :] = g_space[i], m_space[j]
grid = grid.reshape((m_space.shape[0] * g_space.shape[0], 2))

# prob dont need those two, later done in for loop
#g = grid[input_num, 0]
#m = grid[input_num, 1]


'''Set up ALP'''

ngc1275 = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s')
pin = np.diag((1.,1.,0.)) * 0.5
lambda_max = 35.
lambda_min = 0.7
kL = 2. * np.pi / lambda_max
kH = 2. * np.pi / lambda_min


'''Energy binning related stuff goes here'''
log10MeV = np.loadtxt(cwd+'/energy_bins.dat')
full_length = log10MeV.shape[0]
half_length = int(np.ceil(full_length / 2))
# if part == 1:
#     log10MeV = log10MeV[0:half_length]
# elif part == 2:
#     log10MeV = log10MeV[half_length:]
EGeV = np.power(10., log10MeV - 3.)
nbins = EGeV.shape[0]
delta_e = log10MeV[1] - log10MeV[0]    # differences/bin width in log10 E bins
delta_p = delta_e / ppb   # differences/bin width in in log10 prob bins
log_prob_space = np.linspace(log10MeV[0], log10MeV[-1] + delta_e, num=log10MeV.shape[0]*ppb, endpoint=False) - (ppb - 1) / 2 * delta_p
outpath = f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/new_probs/roi_{which_roi}"
try:
    print(f"making directory {outpath}")
    os.mkdir(outpath)
except FileExistsError:
    print(f"{outpath} already exists")
print(log_prob_space.shape)
EGeV_morebins = np.power(10., log_prob_space - 3.)
p_gamma_av = np.zeros((nsim, nbins))
'''Loop over parameters'''
# param_list = [input_num] 
for i in param_list:    # i: index of parameter set g/m+
    # t_0 = time.process_time()
    g = grid[i, 0]
    m = grid[i, 1]
    alp = ALP(m, g)
    m = ModuleList(alp, ngc1275, pin = pin, EGeV = EGeV_morebins)
    m.add_propagation("ICMGaussTurb", 
                      0, # position of module counted from the source. 
                      nsim = nsim, # number of random B-field realizations
                      B0 = B0,  # rms of B field
                      n0 = 39.,  # normalization of electron density
                      n2 = 4.05, # second normalization of electron density, see Churazov et al. 2003, Eq. 4
                      r_abell = 500., # extension of the cluster
                      r_core = 80.,   # electron density parameter, see Churazov et al. 2003, Eq. 4
                      r_core2 = 280., # electron density parameter, see Churazov et al. 2003, Eq. 4
                      beta = 1.2,  # electron density parameter, see Churazov et al. 2003, Eq. 4
                      beta2= 0.58, # electron density parameter, see Churazov et al. 2003, Eq. 4
                      eta = 0.5, # scaling of B-field with electron denstiy
                      kL = kL, # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012 
                      kH = kH,  # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012
                      q = -2.80, # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012
                      seed=seed    # random seed for reproducability, set to None for random seed.
                      )
    m.add_propagation("EBL",1, model = 'dominguez') # EBL attenuation comes second, after beam has left cluster
    m.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS') # finally, the beam enters the Milky Way Field
    p_gamma_x, p_gamma_y, p_a = m.run()
    p_gamma_tot = p_gamma_x + p_gamma_y
    p_orig = p_gamma_tot.copy()
    # np.savetxt(f'/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/new_probs/part_{part}.dat', p_orig)
    del p_a
    del p_gamma_x
    del p_gamma_y
    del p_gamma_tot
    p_orig = p_orig.reshape((nsim, EGeV.shape[0], ppb))
    p_orig = np.average(p_orig, axis=-1)
    # print(p_orig.shape)
    # print(p_gamma_tot.shape)
    # for _sim in range(nsim):
    #     for _bin in range(nbins):
    #         # print(p_gamma_tot[_sim, _bin*ppb:(_bin+1)*ppb])
    #         p_gamma_av[_sim, _bin] = np.average(p_gamma_tot[_sim, _bin*ppb:(_bin+1)*ppb], axis=-1)
    outfile = f"{outpath}/prob_{i:03}.dat"
    try:
        old = np.loadtxt(outfile)
        outdata = np.append(old, p_orig, axis=0)
    except:
        np.savetxt(outfile, p_orig)

    np.savetxt(outfile, outdata)

    '''
    if part == 1:
        print(p_orig.shape)
        np.savetxt('temp_data.dat', p_orig, header='rows: #nsim, columns: #energy bin, '+header)
    elif part == 2:
        half_data = np.loadtxt('temp_data.dat')
        print(half_data.shape)
        half_data = half_data.reshape((nsim, half_data.shape[-1]))
        print(half_data.shape)
        full_data = np.concatenate((half_data, p_orig), axis=1)
        print(full_data.shape)
        prev_data = np.loadtxt(outpath)
        full_data = np.append(prev_data, full_data, axis=0)
        np.savetxt(outfile, full_data, header='rows: #nsim, columns: #energy bin, '+header)
    # outpath_2 = f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/probs/ppb_allbins_{ppb:02}.dat"
    # e_bins_out = f"/nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/probs/ppb_ebins_{ppb:02}.dat"
    # np.savetxt(outpath_2, p_gamma_tot)
    # np.savetxt(e_bins_out, np.log10(EGeV_morebins) + 3, header='in log10(E/MeV)')
    #t_1 = time.process_time()
    #print(f'elapsed time for {nsim} simulations:', t_1 - t_0)
    #print(f'estimated time for 900 parameter combinations:', (t_1 - t_0) * 900)p.savetxt(outpath, p_gamma_av, header=header)
    '''
