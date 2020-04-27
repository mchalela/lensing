import os, platform
import datetime
import numpy as np
import pandas as pd
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM

from .. import gentools 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458. 	# Speed of light (m.s-1)
G    = 6.670e-11   	# Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16 	# 1 pc (m)
Msun = 1.989e30 	# Solar mass (kg)


class Profile(object):

	def __init__(self, data=None, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log', boot_n=0, cosmo=cosmo):
		
		# Create bins...
		self.set_bins(rin_hMpc=rin_hMpc, rout_hMpc=rout_hMpc, bins=bins, space=space)
		nbin = len(self.bins)-1
		self.r_hMpc = 0.5 * (self.bins[:-1] + self.bins[1:])
		self.shear = np.zeros(nbin, dtype=float)
		self.cero = np.zeros(nbin, dtype=float)
		self.shear_error = np.zeros(nbin, dtype=float)
		self.cero_error = np.zeros(nbin, dtype=float)
		self.stat_error = np.zeros(nbin, dtype=float)
		self.N = np.zeros(nbin, dtype=int)

		if data is not None:
			if isinstance(data, pd.DataFrame):
				data = data.to_records()

			# Define some parameters...
			Mpc_scale = self.set_Mpc_scale(dl=data['DL'])
			sigma_critic = self.set_sigma_critic(dl=data['DL'], ds=data['DS'], dls=data['DLS'])

			# Compute distance and ellipticity components...
			dist, theta = gentools.sphere_angular_vector(data['RAJ2000'], data['DECJ2000'],
														data['RA'], data['DEC'], units='deg')
			theta += 90. 
			dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # distance to the lens in Mpc/h
			et, ex = gentools.polar_rotation(data['e1'], data['e2'], np.deg2rad(theta))

			digit = np.digitize(dist_hMpc, bins=self.bins)-1
			m_cal = np.ones(nbin, float)
		
			for i in range(nbin):
				mask = digit==i
				self.N[i] = mask.sum()
				if self.N[i]==0: continue
				weight = data['weight'][mask]/sigma_critic[mask]**2
				m_cal[i] = 1 + np.average(data['m'][mask], weights=weight)

				self.shear[i] = np.average(et[mask]*sigma_critic[mask], weights=weight) / m_cal[i]
				self.cero[i]  = np.average(ex[mask]*sigma_critic[mask], weights=weight) / m_cal[i]

				stat_error_num = np.sum( (0.25*weight*sigma_critic[mask])**2 )
				stat_error_den = weight.sum()**2
				self.stat_error[i] = np.sqrt(stat_error_num/stat_error_den) / m_cal[i]

				if boot_n>0:
					err_t, err_x = self._boot_error(et[mask]*sigma_critic[mask],
													ex[mask]*sigma_critic[mask], 
													weight, boot_n)
					self.shear_error[i] = err_t  / m_cal[i]
					self.cero_error[i] = err_x / m_cal[i]
			
			# Now in units of h*Msun/pc**2
			self.shear /= cosmo.h
			self.cero /= cosmo.h
			self.shear_error /= cosmo.h
			self.cero_error /= cosmo.h
			self.stat_error /= cosmo.h

	def __getitem__(self, key):
		return getattr(self, key)

	def __str__(self):
		try:
			self.__str
		except AttributeError as e:
			p = np.column_stack((self.r_hMpc, self.shear, self.shear_error,
							 self.cero, self.cero_error, self.stat_error, self.N))
			pdf = pd.DataFrame(p, columns=['r_hMpc','shear','shear_error','cero','cero_error','stat_error','N'])
			self.__str = str(pdf)
		finally:
			return self.__str

	def __repr__(self):
		return str(self)

	def set_Mpc_scale(self, dl):
		Mpc_scale = dl*np.deg2rad(1./3600.)
		self.Mpc_scale_mean = Mpc_scale.mean()
		return Mpc_scale

	def set_sigma_critic(self, dl, ds, dls):
		beta = dls/ds
		sigma_critic = cvel**2/(4.*np.pi*G*(dl*1e6*pc)) * (1./beta) * (pc**2/Msun)
		self.beta_mean = beta.mean()
		self.sigma_critic = sigma_critic.mean()
		return sigma_critic

	def set_bins(self, rin_hMpc, rout_hMpc, bins, space):
		if type(bins)==int:
			if space=='log':
				self.bins = np.geomspace(rin_hMpc, rout_hMpc, bins+1)
			else:
				self.bins = np.linspace(rin_hMpc, rout_hMpc, bins+1)
		else:
			self.bins = bins
		self.rin_hMpc  = self.bins[0]
		self.rout_hMpc = self.bins[-1]
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

	def write_to(self, file, header=None, colnames=True, overwrite=False):
		'''Add a header to lensing.shear.Profile output file
		to know the sample parameters used to build it.
		
		 file: 	    (str) Name of output file
		 header:    (dic) Dictionary with parameter cuts. Optional.
	            Example: {'z_min':0.1, 'z_max':0.3, 'odds_min':0.5}
	     colnames:  (bool) Flag to write columns names as first line.
	     		If False, column names are commented.
		'''
		if os.path.isfile(file):
			if overwrite:
				os.remove(file)
			else:
				raise IOError('File already exist. You may want to use overwrite=True.')

		with open(file, 'a') as f:
			f.write('# '+'-'*48+'\n')
			f.write('# '+'\n')
			f.write('# Lensing profile '+'\n')
			if header is not None:
				for key, value in list(header.items()):
					f.write('# '+key.ljust(14)+' = '+str(value) +'\n')

			f.write('# '+'\n')
			f.write('# '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'\n')
			f.write('# '+'-'*48+'\n')

			C = ' ' if colnames else '# '
			f.write(C+'r_hMpc  shear  shear_error  cero  cero_error  stat_error  N \n')
			p = np.column_stack((self.r_hMpc, self.shear, self.shear_error,
								 self.cero, self.cero_error, self.stat_error, self.N))
			np.savetxt(f, p, fmt=['%12.6f']*6+['%8i'])		
		return None

	@gentools.classonly
	def read_profile(self, file, colnames=True):
		''' Read profile written with write_to() method
		'''

		with open(file,'r') as f:
			for i, line in enumerate(f):
				if not line.startswith('#'): break
			f.seek(0,0)
			pf = np.genfromtxt(f, dtype=None, names=colnames, skip_header=i)

		p = Profile()
		p.r_hMpc = pf['r_hMpc']
		p.shear = pf['shear']
		p.cero = pf['cero']
		p.shear_error = pf['shear_error']
		p.cero_error = pf['cero_error']
		p.stat_error = pf['stat_error']
		p.N = pf['N']
		return p