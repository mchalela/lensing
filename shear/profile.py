import os, platform
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from lensing import gentools

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458. 	# Speed of light (m.s-1)
G    = 6.670e-11   	# Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16 	# 1 pc (m)
Msun = 1.989e30 	# Solar mass (kg)


class Profile(object):

	def __init__(self, cat, rin_Mpc=0.1, rout_Mpc=10., bins=10, space='log', cosmo=cosmo, boot_flag=True, boot_n=100):
		# Define some parameters...
		self.set_Mpc_scale(dl=cat['DL'].to_numpy())
		self.set_sigma_critic(dl=cat['DL'].to_numpy(),
							  ds=cat['DS'].to_numpy(),
							  dls=cat['DLS'].to_numpy())

		# Compute distance and ellipticity components...
		dist, theta = gentools.sphere_angular_vector(cat['RAJ2000'].to_numpy(), cat['DECJ2000'].to_numpy(),
													cat['RA'].to_numpy(), cat['DEC'].to_numpy(), units='deg')
		theta += 90. #np.pi/2.
		dist_Mpc = dist*3600.*self.Mpc_scale # distance to the lens in Mpc
		et, ex = gentools.polar_rotation(cat['e1'].to_numpy(), cat['e2'].to_numpy(), np.deg2rad(theta))

		# Create bins...
		if type(bins)==int:
			if space=='log':
				self.bins = np.geomspace(rin_Mpc, rout_Mpc, bins+1)
			else:
				self.bins = np.linspace(rin_Mpc, rout_Mpc, bins+1)
		else:
			self.bins = bins
			rin_Mpc  = self.bins[0]
			rout_Mpc = self.bins[-1]

		nbin = len(self.bins)-1
		digit = np.digitize(dist_Mpc, bins=self.bins)-1
	
		self.r_Mpc = 0.5 * (self.bins[:-1] + self.bins[1:])
		self.shear = np.zeros(nbin, dtype=float)
		self.cero = np.zeros(nbin, dtype=float)
		self.shear_error = np.zeros(nbin, dtype=float)
		self.cero_error = np.zeros(nbin, dtype=float)
		self.stat_error = np.zeros(nbin, dtype=float)

		m_cal = np.ones(nbin, float)
		self.N = np.zeros(nbin, int)

		for i in range(nbin):
			mask = digit==i
			self.N[i] = mask.sum()
			if self.N[i]==0: continue
			weight = cat['weight'][mask].to_numpy()/self.sigma_critic[mask]**2
			m_cal[i] = 1 + np.average(cat['m'][mask].to_numpy(), weights=weight)

			self.shear[i] = np.average(et[mask]*self.sigma_critic[mask], weights=weight) / m_cal[i]
			self.cero[i]  = np.average(ex[mask]*self.sigma_critic[mask], weights=weight) / m_cal[i]

			stat_error_num = np.sum( (0.25*weight*self.sigma_critic[mask])**2 )
			stat_error_den = weight.sum()**2
			self.stat_error[i] = np.sqrt(stat_error_num/stat_error_den) / m_cal[i]

			if boot_flag:
				err_t, err_x = self._boot_error(et[mask]*self.sigma_critic[mask],
												ex[mask]*self.sigma_critic[mask], 
												weight, boot_n)
				self.shear_error[i] = err_t  / m_cal[i]
				self.cero_error[i] = err_x / m_cal[i]

		
	def set_Mpc_scale(self, dl):
		self.Mpc_scale = dl*np.deg2rad(1./3600.)
		return None

	def set_sigma_critic(self, dl, ds, dls):
		self.beta = dls/ds
		self.sigma_critic = cvel**2/(4.*np.pi*G*(dl*1e6*pc)) * (1./self.beta) * (pc**2/Msun)
		return None

	def _boot_error(self, shear, cero, weight, nboot):
		index=np.arange(len(shear))
		with NumpyRNGContext(seed=1):
			bootresult = bootstrap(index, nboot)
		index_boot  = bootresult.astype(int)
		shear_boot  = shear[index_boot]	
		cero_boot   = cero[index_boot]	
		weight_boot = weight[index_boot]	
		shear_means = np.average(shear_boot, weights=weight_boot, axis=1)
		cero_means  = np.average(cero_boot, weights=weight_boot, axis=1)
		return np.std(shear_means), np.std(cero_means)


