import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ; plt.ion()
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from scipy import fftpack, ndimage

from . import Map, KappaMap
from .. import gentools, Shear

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)




class Profile(object):

	def __init__(self, rin=0.1, rout=10., nbins=10, space='log'):
		

		self.bins = gentools.make_bins(rin, rout, nbins=nbins, space=space)
		self.nbins = nbins
		self.rin  = self.bins[0]
		self.rout = self.bins[-1]

		# Initialize
		self.r = None
		self.sigmaE = None
		self.sigmaB = None
		self.sigmaE_error = None
		self.sigmaB_error = None
		self.deltaSigmaE = None
		self.deltaSigmaB = None
		self.deltaSigmaE_error = None
		self.deltaSigmaB_error = None

	def __getitem__(self, key):
		return getattr(self, key)

	def __str__(self):
		try:
			self.__str
		except AttributeError as e:
			p = np.column_stack((self.r, self.sigmaE, self.sigmaE_error,
				self.sigmaB, self.sigmaB_error,
				self.deltaSigmaE, self.deltaSigmaE_error,
				self.deltaSigmaB, self.deltaSigmaB_error))
			pdf = pd.DataFrame(p, columns=['r','sigmaE','sigmaE_error','sigmaB','sigmaB_error',
				'deltaSigmaE','deltaSigmaE_error','deltasigmaB','deltaSigmaB_error'])
			self.__str = str(pdf)
		finally:
			return self.__str

	def __repr__(self):
		return str(self)

class RadialProfile(Profile):

	def __init__(self, kappa_map, centre=None, rin=0.1, rout=10., nbins=10, sigma=None, space='log'):
		
		#if not isinstance(kappa_map, (Kappa.Map, Kappa.CompressedMap)):
		#	raise TypeError('kappa_map must be a Kappa Map instance.')

		super().__init__(rin, rout, nbins, space)

		if centre is None:
			centre = np.array([0., 0.])
		
		self.centre = centre

		pf = self._profile(kappa_map, self.centre, sigma)

		self.r = pf['r']
		self.sigmaE = pf['sE_pf']
		self.sigmaB = pf['sB_pf']
		self.sigmaE_error = pf['sE_err']
		self.sigmaB_error = pf['sB_err']
		self.deltaSigmaE = pf['dsE_pf']
		self.deltaSigmaB = pf['dsB_pf']
		self.deltaSigmaE_error = pf['dsE_err']
		self.deltaSigmaB_error = pf['dsB_err']


	def _profile(self, kmp, centre, sigma):

		if sigma is not None:
			skmp = kmp.gaussian_filter(sigma)
			massE = skmp.real.copy()
			massB = skmp.imag.copy()
		else:
			massE = kmp.kappa.real.copy()
			massB = kmp.kappa.imag.copy()

		massE *= kmp.shear_map.sigma_critic
		massB *= kmp.shear_map.sigma_critic

		px = kmp.px - centre[0]
		py = kmp.py - centre[1]

		# Flatten the arrays and compute the distance to the centre
		massE = massE.flatten()
		massB = massB.flatten()
		x = px.flatten()
		y = py.flatten()
		dist = np.sqrt(x**2 + y**2)

		# Compute the profile
		digit = np.digitize(dist, bins=self.bins) - 1

		bins_centres = []
		sE_pf = []
		sB_pf = []
		sE_err = []
		sB_err = []
		dsE_pf = []
		dsB_pf = []
		dsE_err = []
		dsB_err = []
		for i in range(len(self.bins)-1):
			imask = digit==i
			innermask = digit<i
			bins_centres.append( 0.5*(self.bins[i]+self.bins[i+1]) )
			
			sE_pf.append( np.average(massE[imask]) )
			sB_pf.append( np.average(massB[imask]) )
			sE_err.append( np.std(massE[imask]) )
			sB_err.append( np.std(massB[imask]) )

			dsE_pf.append( np.average(massE[innermask]) - sE_pf[i] )
			dsB_pf.append( np.average(massB[innermask]) - sB_pf[i] )
			dsE_err.append( np.std(massE[innermask]) + sE_err[i] )
			dsB_err.append( np.std(massB[innermask]) + sB_err[i] )

		sE_pf = np.asarray(sE_pf)
		sB_pf = np.asarray(sB_pf)
		sE_err = np.asarray(sE_err)
		sB_err = np.asarray(sB_err)
		dsE_pf = np.asarray(dsE_pf)
		dsB_pf = np.asarray(dsB_pf)
		dsE_err = np.asarray(dsE_err)
		dsB_err = np.asarray(dsB_err)

		bins_centres = np.asarray(bins_centres)

		x = {}
		x['r'] = bins_centres
		x['sE_pf'] = sE_pf
		x['sE_err'] = sE_err
		x['sB_pf'] = sB_pf
		x['sB_err'] = sB_err
		x['dsE_pf'] = dsE_pf
		x['dsE_err'] = dsE_err
		x['dsB_pf'] = dsB_pf
		x['dsB_err'] = dsB_err
		return x

	def QuickPlot(self):
		import matplotlib.pyplot as plt

		plt.plot(self.r, self.sigmaE)
