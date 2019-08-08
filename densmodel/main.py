import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
import multiprocessing

import emcee
#import corner
from astropy import units
from astropy.cosmology import Planck15 #, LambdaCDM
from lensing import shear, densmodel



name = 'bin4_COMB'
z=0.25
offset = 0.4	# in Mpc/h
#path = '/home/martin/Documentos/Doctorado/Lentes/lensing/redMapper_test/redMapper_COMB.profile'
path = '/home/mchalela/redMaPPer/redMapper_COMB.profile'
profile = shear.Profile.read_profile(path)
rbins = profile['r_hMpc']		# R in Mpc/h
shear_obs = profile['shear'] / 0.7	# Density contrast in h*Msun/pc2
shear_err = profile['shear_error'] / 0.7
args_fix_off=(offset, z, rbins, shear_obs, shear_err)
args_fix_none=(z, rbins, shear_obs, shear_err)
densmodel.mcmc.mcmc_create_samples(args=args_fix_off, ndim=2, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
densmodel.mcmc.mcmc_create_samples(args=args_fix_none, ndim=3, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.3.'+name+'.txt'