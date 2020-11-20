from fermipy.gtanalysis import GTAnalysis
import os
import yaml
path_to_conda = os.environ['CONDA_PREFIX']
sc_path = '/nfs/astrop/n1/kuhlmann/NGC_1275/raw_data_roi_15/L2009090432059398E3A906_SC00.fits'
cwd = os.getcwd()
with open(cwd+'/config_local.yaml', 'r') as i:
    config = yaml.safe_load(i)

config['fileio']['workdir'] =  cwd+'/fits'
config['fileio']['outdir'] = cwd+'/fits'
config['fileio']['logfile'] = cwd+'/fits/fermipy.log'
config['data']['ltcube'] = cwd+'/fits/ltcube_00.fits'
config['data']['scfile'] = sc_path
config['model']['galdiff'] = path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
config['model']['isodiff'] = path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'
config['logging']['verbosity'] = 4

source = config['selection']['target']

with open(cwd+'/config_modified.yaml', 'w') as o:
    yaml.dump(config, o)
prefix = 'simulation_base'
gta = GTAnalysis('config_modified.yaml', logging={'verbosity' : 4},\
		 model={'galdiff': path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits', \
		        'isodiff': path_to_conda+'/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_SOURCE_V3_v1.txt'})
gta.setup()
# gta.write_roi(prefix+'_setup.npy')
#gta = GTAnalysis.create('/fits/'+prefix+'.npy', config='config_modified.yaml')
#print(gta.like())
dloglike=1
i=0

while dloglike > 1e-5:
    print('optimizing...')
    o = gta.optimize()
    dloglike = o['dloglike'] / o['loglike0']
    i += 1
    if i >=10:
        print('10th iteration, will stop now')
        break
gta.free_sources(free=False)
gta.free_sources(distance=3.0, pars='norm')
# gta.free_sources(minmax_TS=[None,10],free=False,pars='norm')
# gta.free_sources(minmax_npred=[10,100],free=False,pars='norm')
gta.free_source('isodiff')
gta.free_source('galdiff')
gta.free_source('4FGL J0319.8+4130')
gta.free_sources(minmax_ts=[100, None], pars='norm')
fit_result = gta.fit(optimizer='NEWMINUIT', reoptimize=True)
gta.write_roi(prefix+'_roi')
gta.write_roi(cwd+'/fits/base.npy')
sed = gta.sed('4FGL J0319.8+4130', make_plots=True, prefix=prefix, free_background=False)

