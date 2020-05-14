import os, platform
import datetime
import numpy as np
import pandas as pd
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from joblib import Parallel, delayed

from .. import gentools
from ..LensCat import CompressedCatalog, ExpandedCatalog

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458. 	# Speed of light (m.s-1)
G    = 6.670e-11   	# Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16 	# 1 pc (m)
Msun = 1.989e30 	# Solar mass (kg)


# PICKABLE functions ----------------------------------------
def _profile_per_bin(i, dict_per_bin):

	mask = dict_per_bin['digit']==i

	m = dict_per_bin['m'][mask]
	weight = dict_per_bin['weight'][mask]
	et = dict_per_bin['et'][mask]
	ex = dict_per_bin['ex'][mask]
	sigma_critic = dict_per_bin['sigma_critic'][mask]
	
	N_i = mask.sum()
	if N_i==0: return [0.0]*6
	
	w = weight/sigma_critic**2
	
	accum_w_i = np.sum(w) 
	m_cal_num_i = np.sum(w*(1+m))
	shear_i = np.sum(et*sigma_critic*w)
	cero_i = np.sum(ex*sigma_critic*w)
	stat_error_num_i = np.sum( (0.25*w*sigma_critic)**2 )

	return shear_i, cero_i, accum_w_i, m_cal_num_i, stat_error_num_i, N_i

	#if self.boot_n>0:
	#	err_t, err_x = self._boot_error(et*sigma_critic, ex*sigma_critic, w, self.boot_n)
	#	self.shear_error[i] = err_t  / m_cal[i]
	#	self.cero_error[i] = err_x / m_cal[i]


def _profile_per_lens(j, dict_per_lens):
	print(j)
	data_L = dict_per_lens['data_L']
	data_S = dict_per_lens['data_S']
	bins = dict_per_lens['bins']
	#cosmo = dict_per_lens['cosmo']

	dL = data_L.iloc[j]
	dS = data_S.loc[dL['CATID']]

	DD = gentools.compute_lensing_distances(zl=dL['Z'], zs=dS['Z_B'],
		dist=['DL', 'DS', 'DLS'], precomputed=True)#, cosmo=self.cosmo)

	Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
	sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])

	# Compute distance and ellipticity components...
	dist, theta = gentools.sphere_angular_vector(dS['RAJ2000'], dS['DECJ2000'],
												dL['RA'], dL['DEC'], units='deg')
	theta += 90. 
	dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h
	et, ex = gentools.polar_rotation(dS['e1'], dS['e2'], np.deg2rad(theta))

	digit = np.digitize(dist_hMpc, bins=bins)-1
		
	dict_per_bin = {'m': dS['m'], 'weight': dS['weight'],
				'digit': digit, 'sigma_critic': sigma_critic,
				'et': et, 'ex': ex}

	shear_j, cero_j = [], []
	accum_w_j, m_cal_num_j = [], []
	stat_error_num_j, N_j = [], []

	for i in range(len(bins)):
		pf = _profile_per_bin(i, dict_per_bin)
		shear_j += [pf[0]]
		cero_j += [pf[1]]
		accum_w_j += [pf[2]]
		m_cal_num_j += [pf[3]]
		stat_error_num_j += [pf[4]]
		N_j += [pf[5]]

	shear_j, cero_j = np.array(shear_j), np.array(cero_j)
	accum_w_j, m_cal_num_j = np.array(accum_w_j), np.array(m_cal_num_j)
	stat_error_num_j, N_j = np.array(stat_error_num_j), np.array(N_j)

	return shear_j, cero_j, accum_w_j, m_cal_num_j, stat_error_num_j, N_j


class ExpandedProfile(object):

	def __init__(self, data=None, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log', boot_n=0, cosmo=cosmo,
		back_dz=0., precompute_distances=True, reduce=True, njobs=1):
		
		#if not isinstance(cat, ExpandedCatalog):
		#	raise TypeError('cat must be a LensCat.ExpandedCatalog catalog.')

		self.cosmo = cosmo
		self.boot_n = boot_n
		self.reduce_flag = reduce
		self.njobs = njobs

		self.bins = gentools.make_bins(rin=rin_hMpc, rout=rout_hMpc, bins=bins, space=space)
		self.nbin = len(self.bins)-1
		self.rin_hMpc  = self.bins[0]
		self.rout_hMpc = self.bins[-1]
		self.r_hMpc = 0.5 * (self.bins[:-1] + self.bins[1:])

		self.shear_error = np.zeros(self.nbin, dtype=float)
		self.cero_error = np.zeros(self.nbin, dtype=float)

		self._profile(data=data)

		'''
		# Now in units of h*Msun/pc**2
		self.shear /= self.cosmo.h
		self.cero /= self.cosmo.h
		self.shear_error /= self.cosmo.h
		self.cero_error /= self.cosmo.h
		self.stat_error /= self.cosmo.h
		'''

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

	def _reduce(self, pf):

		shear = np.array( [pf[_][0] for _ in range(self.nbin)] )
		cero = np.array( [pf[_][1] for _ in range(self.nbin)] )
		accum_w = np.array( [pf[_][2] for _ in range(self.nbin)] )
		m_cal_num = np.array( [pf[_][3] for _ in range(self.nbin)] )
		stat_error_num = np.array( [pf[_][4] for _ in range(self.nbin)] )
		N = np.array( [pf[_][5] for _ in range(self.nbin)] )

		self.m_cal = m_cal_num / accum_w
		self.shear = (shear/accum_w) / self.m_cal
		self.cero = (cero / accum_w) / self.m_cal
		self.stat_error = np.sqrt(stat_error_num/accum_w**2) / self.m_cal
		self.N = N

	def _profile(self, data):
		''' Computes profile for ExpandedCatalog
		'''
		DD = gentools.compute_lensing_distances(zl=data['Z'], zs=data['Z_B'],
				dist=['DL', 'DS', 'DLS'], precomputed=True, cosmo=self.cosmo)
		
		Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
		sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])

		# Compute distance and ellipticity components...
		dist, theta = gentools.sphere_angular_vector(data['RAJ2000'], data['DECJ2000'],
													data['RA'], data['DEC'], units='deg')
		theta += 90. 
		dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h
		et, ex = gentools.polar_rotation(data['e1'], data['e2'], np.deg2rad(theta))

		digit = np.digitize(dist_hMpc, bins=self.bins)-1
			
		dict_per_bin = {'m': data['m'], 'weight': data['weight'],
					'digit': digit, 'sigma_critic': sigma_critic,
					'et': et, 'ex': ex}

		print('GO!!')
		with Parallel(n_jobs=self.njobs) as parallel:
			delayed_fun = delayed(_profile_per_bin)
			pf = parallel(delayed_fun(i, dict_per_bin) for i in range(self.nbin))

		self._reduce(pf)


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



class CompressedProfile(object):

	def __init__(self, data_L, data_S, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log', boot_n=0, cosmo=cosmo,
		back_dz=0., precompute_distances=True, reduce=True, njobs=1):
		
		#if not isinstance(cat, (CompressedCatalog, ExpandedCatalog)):
		#	raise TypeError('cat must be a LensCat catalog.')

		self.cosmo = cosmo
		self.boot_n = boot_n
		self.reduce_flag = reduce
		self.njobs = njobs

		self.bins = gentools.make_bins(rin=rin_hMpc, rout=rout_hMpc, bins=bins, space=space)
		self.nbin = len(self.bins)-1
		self.rin_hMpc  = self.bins[0]
		self.rout_hMpc = self.bins[-1]
		self.r_hMpc = 0.5 * (self.bins[:-1] + self.bins[1:])

		self.shear_error = np.zeros(self.nbin, dtype=float)
		self.cero_error = np.zeros(self.nbin, dtype=float)

		if data_S.index.name is not 'CATID':
			data_S_indexed = data_S.set_index('CATID')
		self._profile(data_L=data_L, data_S=data_S_indexed)

		'''
		# Now in units of h*Msun/pc**2
		self.shear /= self.cosmo.h
		self.cero /= self.cosmo.h
		self.shear_error /= self.cosmo.h
		self.cero_error /= self.cosmo.h
		self.stat_error /= self.cosmo.h
		'''

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

	def _reduce(self, pf):

		shear = np.sum( [pf[_][0] for _ in range(len(pf))] )
		cero = np.sum( [pf[_][1] for _ in range(len(pf))] )
		accum_w = np.sum( [pf[_][2] for _ in range(len(pf))] )
		m_cal_num = np.sum( [pf[_][3] for _ in range(len(pf))] )
		stat_error_num = np.sum( [pf[_][4] for _ in range(len(pf))] )
		N = np.sum( [pf[_][5] for _ in range(len(pf))] )

		self.m_cal = m_cal_num / accum_w
		self.shear = (shear/accum_w) / self.m_cal
		self.cero = (cero / accum_w) / self.m_cal
		self.stat_error = np.sqrt(stat_error_num/accum_w**2) / self.m_cal
		self.N = N

	def _profile(self, data_L, data_S):
		''' Computes profile for CompressedCatalog
		'''
		dict_per_lens = {'data_L': data_L, 'data_S': data_S, 'bins': self.bins}

		# Calcular perfiles por grupos
		print('GO!!')
		with Parallel(n_jobs=self.njobs) as parallel:
			delayed_fun = delayed(_profile_per_lens)
			pf = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))

		# Reducir
		self._reduce(pf)
		
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