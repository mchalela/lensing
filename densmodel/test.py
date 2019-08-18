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
logMstar= np.log10(6.4e11)
vdisp = 250.
r = np.geomspace(0.01, 3., 40)
#bcg = DN.BCG_with_M200(r, logM200)
bcg = DN.BCG(r, logMstar)
nfw = DN.NFW(r, logM200)
sis = DN.SIS(r, vdisp)
shear = bcg + nfw
#shear = sis.copy()


shear *= 1+np.random.normal(0., 0.2, r.shape)
shear_err = 0.2*shear

DNM = densmodel.DensityModels(z=z, cosmo=Planck15)
#bcg_init = DNM.BCG_with_M200(logM200_0=13.)
bcg_init = DNM.BCG(logMstar_0=12.)
nfw_init = DNM.NFW(logM200_0=13.)
#shalo_init = DNM.SecondHalo(logM200_0=13.)
shear_init = DNM.AddModels([bcg_init, nfw_init])#, shalo_init])
#shear_init2 = nfw_init.copy()#, shalo_init])
#shear_init = DNM.SIS(disp_0=500.)

start_params = [12., 13.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
out_min = fitter.Minimize(method='GLL')
_fitter_to_model_params(shear_init, out_min.param_values)

'''
start_params = [11., 14.]
fitter = densmodel.Fitter(r, shear, shear_err, shear_init, start_params)
nwalkers=4; steps=300; ndim=len(shear_init.parameters)
sampler, samples_file = fitter.MCMC(method='GLP',nwalkers=nwalkers, steps=steps, sample_name='test')
'''

cvel=299792458;   # Speed of light (m.s-1)
G= 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc= 3.085678e16; # 1 pc (m)
Msun=1.989e30 # Solar mass (kg)

def SIS_stack_fit(R,D_Sigma,err):
	def chi_red(ajuste,data,err,gl):
		BIN=len(data)
		chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
		return chi
	# R en Mpc, D_Sigma M_Sun/pc2
	def sis_profile_sigma(R,sigma):
		Rm=R*1.e6*pc
		return (((sigma*1.e3)**2)/(2.*G*Rm))*(pc**2/Msun)
	
	sigma,err_sigma_cuad=scipy.optimize.curve_fit(sis_profile_sigma,R,D_Sigma,sigma=err,absolute_sigma=True)
	
	ajuste=sis_profile_sigma(R,sigma)
	
	chired=chi_red(ajuste,D_Sigma,err,1)	
	
	xplot=R.copy() #np.arange(0.001,R.max()+1.,0.001)
	yplot=sis_profile_sigma(xplot,sigma)
	
	return sigma[0],np.sqrt(err_sigma_cuad)[0][0],chired,xplot,yplot

old_fit = SIS_stack_fit(r, shear, shear_err)


#plt.plot(r, sis, 'k-')
plt.plot(r, shear, 'k.')
plt.errorbar(r, shear, yerr=shear_err, fmt='None', ecolor='k')
#plt.plot(r, nfw, 'r--')
plt.loglog()
plt.plot(r, shear_init(r), 'r-', lw=2)
plt.plot(old_fit[-2], old_fit[-1],'C0--')
