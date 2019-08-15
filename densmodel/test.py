import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
#import multiprocessing
from astropy.modeling.fitting import _fitter_to_model_params
import emcee
import corner
from astropy import units
from astropy.cosmology import Planck15 , LambdaCDM
from astropy.modeling import models, fitting
from lensing import shear, densmodel


import scipy.optimize

# Toy Model

z=0.2
DN = densmodel.Density(z=z, cosmo=Planck15)
logM200 = np.log10(2.3e14)
logMstar= np.log10(2.4e11)
r = np.geomspace(0.01, 1., 20)
#bcg = DN.BCG_with_M200(r, logM200)
bcg = DN.BCG(r, logMstar)
nfw = DN.NFW(r, logM200)
shear = bcg + nfw


shear *= 1+np.random.normal(0., 0.1, r.shape)
shear_err = 0.1*shear

DNM = densmodel.DensityModels(z=z, cosmo=Planck15)
#bcg_init = DNM.BCG_with_M200(logM200_0=13.)
bcg_init = DNM.BCG(logMstar_0=12.)
nfw_init = DNM.NFW(logM200_0=13.)
#shalo_init = DNM.SecondHalo(logM200_0=13.)
shear_init = DNM.AddModels([bcg_init, nfw_init])#, shalo_init])
#shear_init = nfw_init.copy()#, shalo_init])


start_params = [12., 13.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
out_min = fitter.Minimize(method='GLP')
_fitter_to_model_params(shear_init, out_min.x)


start_params = [11., 14.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
nwalkers=4; steps=300; ndim=len(shear_init.parameters)
sampler, samples_file = fitter.MCMC(method='GLP',nwalkers=nwalkers, steps=steps, sample_name='test')





plt.plot(r, shear, 'k.')
plt.errorbar(r, shear, yerr=shear_err, fmt='None', ecolor='k')
#plt.plot(r, nfw, 'r--')
plt.loglog()
plt.plot(r, shear_init(r), 'm--')
