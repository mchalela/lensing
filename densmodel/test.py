import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
#import multiprocessing

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
#shear = bcg + nfw
shear = nfw.copy()

shear *= 1+np.random.normal(0., 0.1, r.shape)
shear_err = 0.1*shear

DNM = densmodel.DensityModels(z=z, cosmo=Planck15)
#bcg_init = DNM.BCG_with_M200(logM200_0=13.)
bcg_init = DNM.BCG(logMstar_0=12.)
nfw_init = DNM.NFW(logM200_0=13.)
#shalo_init = DNM.SecondHalo(logM200_0=13.)
#shear_init = DNM.AddModels([bcg_init, nfw_init])#, shalo_init])
shear_init = nfw_init#, shalo_init])

np.random.seed(0)
x = np.linspace(-5., 5., 200)
y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
y += np.random.normal(0., 0.2, x.shape)
yerr = 0.2
g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
start_params = [1, 0, 1]
fitter = densmodel.Fitter(x, y, yerr, g_init, start_params)
nwalkers=20; steps=300; ndim=len(g_init.parameters)
samples_file = fitter.MCMC(nwalkers=nwalkers, steps=steps, sample_name='gausstest', threads=2)

# Ajuste con Posterior Likelihood
ploglike = GaussianLogPosterior(r, shear, shear_err, shear_init)
neg_ploglike = lambda x: -ploglike(x)
start_params = [15., 15.]
opt = scipy.optimize.minimize(neg_ploglike, start_params, method="L-BFGS-B", tol=1.e-10)
print opt.x

# Ajuste con Likelihood
loglike = GaussianLogLikelihood(r, shear, shear_err, shear_init)
neg_loglike = lambda x: -loglike(x)
start_params = [11., 13.]
opt = scipy.optimize.minimize(neg_loglike, start_params, method="L-BFGS-B", tol=1.e-10)
print opt.x
start_params = [12.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
out_min = fitter.Minimize(method='GLL')
_fitter_to_model_params(shear_init, out_min.x)


start_params = [12.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
nwalkers=20; steps=300; ndim=len(shear_init.parameters)
samples_file = fitter.MCMC(model=shear_init, nwalkers=nwalkers, steps=steps, sample_name='test', threads=2)
samplesf = np.loadtxt(samples_file)
sampler_chain = samplesf.reshape((nwalkers,steps,ndim))

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
axes[0].plot(sampler_chain[:, :, 0].T, color="k", alpha=0.4)
#axes[0].axhline(logm_true, color="g", lw=2)
axes[0].set_ylabel("log-mass")

burn_in_step = 100  # based on a rough look at the walker positions above
log_samples = sampler_chain[:, burn_in_step:, :].reshape((-1, ndim))

samples = log_samples.copy()
samples[:, 0] = 10**(log_samples[:, 0])
M_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
M_fit = M_mcmc[0][0] ; M_perr = M_mcmc[0][1] ; M_merr = M_mcmc[0][2]
fig = corner.corner(log_samples,
                    labels=["$\mathrm{log}M_{200}$"],
                    truths=[np.log10(M_fit)])
######################################################
#import profiles_fit
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

H        = cosmo.H(z).value/(1.0e3*pc) #H at z_pair s-1 
roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
roc_mpc  = roc*((pc*1.0e6)**3.0)


nfw_fit    = profiles_fit.NFW_stack_fit(r/cosmo.h,shear*cosmo.h,shear_err*cosmo.h,z,roc)
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


plt.plot(r, shear, 'k.')
plt.errorbar(r, shear, yerr=shear_err, fmt='None', ecolor='k')
#plt.plot(r, nfw, 'r--')
plt.loglog()
plt.plot(r, shear_init(r), 'm--')
plt.plot(x2*cosmo.h, y2/cosmo.h, 'g:')