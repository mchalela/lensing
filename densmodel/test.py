import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
#import multiprocessing

import emcee
#import corner
from astropy import units
from astropy.cosmology import Planck15 #, LambdaCDM
from astropy.modeling import models
from lensing import shear, densmodel

import scipy.optimize

# Toy Model

z=0.2
DN = densmodel.Density(z=z, cosmo=Planck15)
logM200 = np.log10(2.3e14)
r = np.geomspace(0.01, 1., 100)
bcg = DN.BCG_with_M200(r, logM200)
nfw = DN.NFW(r, logM200)
shear = bcg + nfw

shear *= 1+np.random.normal(0., 0.2, r.shape)
shear_err = 0.2*shear

DNM = densmodel.DensityModels(z=z, cosmo=Planck15)
bcg_init = DNM.BCG_with_M200(logM200_0=13.)
nfw_init = DNM.NFW(logM200_0=13.)
shalo_init = DNM.SecondHalo(logM200_0=13.)
shear_init = DNM.AddModels([bcg_init, nfw_init, shalo_init])


plt.plot(r, shear, 'k.')
plt.plot(r, bcg+nfw, 'r--')
plt.errorbar(r, shear, yerr=shear_err, fmt='None', ecolor='k')
plt.loglog()


loglike = GaussianLogLikelihood(r, shear, shear_err, shear_init)
neg_loglike = lambda x: -loglike(x)

start_params = [15., 13.]
opt = scipy.optimize.minimize(neg_loglike, start_params, method="L-BFGS-B", tol=1.e-10)

fit_pars = opt.x
_fitter_to_model_params(loglike.model, fit_pars)

plt.plot(r, loglike.model(r), 'm--')