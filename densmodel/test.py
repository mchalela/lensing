import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
#import multiprocessing

import emcee
#import corner
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
r = np.geomspace(0.01, 1., 100)
#bcg = DN.BCG_with_M200(r, logM200)
bcg = DN.BCG(r, logMstar)
nfw = DN.NFW(r, logM200)
#shear = bcg + nfw
shear = nfw.copy()

shear *= 1+np.random.normal(0., 0.05, r.shape)
shear_err = 0.2*shear

DNM = densmodel.DensityModels(z=z, cosmo=Planck15)
#bcg_init = DNM.BCG_with_M200(logM200_0=13.)
bcg_init = DNM.BCG(logMstar_0=12.)
nfw_init = DNM.NFW(logM200_0=13.)
#shalo_init = DNM.SecondHalo(logM200_0=13.)
#shear_init = DNM.AddModels([bcg_init, nfw_init])#, shalo_init])
shear_init = nfw_init#, shalo_init])



# Ajuste con Posterior Likelihood
ploglike = GaussianLogPosterior(r, shear, shear_err, shear_init)
neg_ploglike = lambda x: -ploglike(x)
start_params = [14., 15.]
opt = scipy.optimize.minimize(neg_ploglike, start_params, method="L-BFGS-B", tol=1.e-10)
print opt.x

# Ajuste con Likelihood
loglike = GaussianLogLikelihood(r, shear, shear_err, shear_init)
neg_loglike = lambda x: -loglike(x)
start_params = [12., 13.]
opt = scipy.optimize.minimize(neg_loglike, start_params, method="L-BFGS-B", tol=1.e-10)
print opt.x

# Ajuste con Astropy fitting
fitter = fitting.LevMarLSQFitter()
out= fitter(shear_init, r, shear, weights=1./shear_err ,maxiter=100, acc=1e-2, epsilon=1.e-2)
print out

# Ajuste con curve_fit como antes

def func(r, ms, m200):
	return DN.BCG(r, ms)+DN.NFW(r, m200)
optc = scipy.optimize.curve_fit(DN.NFW, r, shear, sigma=shear_err, bounds=([10.] , [16.]))#,absolute_sigma=True)
print optc


fit_pars = opt.x
_fitter_to_model_params(loglike.model, fit_pars)



######################################################

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

H        = cosmo.H(z).value/(1.0e3*pc) #H at z_pair s-1 
roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
roc_mpc  = roc*((pc*1.0e6)**3.0)


nfw_fit        = profiles_fit.NFW_stack_fit(r/cosmo.h,shear*cosmo.h,shear_err*cosmo.h,z,roc)
c          = nfw_fit[5]
CHI_nfw    = nfw_fit[2]
RS         = nfw_fit[0]/c
R200       = nfw_fit[0]
error_R200 = nfw_fit[1]
x2         = nfw_fit[3]
y2         = nfw_fit[4]

M200_NFW   = (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)
e_M200_NFW = ((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*error_R200
print np.log10(M200_NFW*cosmo.h)


############


#plt.plot(r, shear, 'k.')
#plt.errorbar(r, shear, yerr=shear_err, fmt='None', ecolor='k')
plt.plot(r, nfw, 'r--')
plt.loglog()
plt.plot(r, loglike.model(r), 'm--')
plt.plot(x2*cosmo.h, y2/cosmo.h, 'g:')