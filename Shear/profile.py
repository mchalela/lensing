import os, platform
import datetime
import numpy as np
import pandas as pd
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from joblib import Parallel, delayed

from .. import gentools

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458. 	# Speed of light (m.s-1)
G    = 6.670e-11   	# Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16 	# 1 pc (m)
Msun = 1.989e30 	# Solar mass (kg)

np.random.seed(0)

# PICKABLE functions =================================================================
def _profile_per_bin(i, dict_per_bin):

	nboot = dict_per_bin['nboot']

	#mask = dict_per_bin['digit']==i
	#N_i = mask.sum()
	mask = np.where(dict_per_bin['digit']==i)[0]
	N_i = len(mask)
	if N_i==0: 
		x = {'shear_i': np.zeros(nboot+1), 'cero_i': np.zeros(nboot+1),
			 'accum_w_i': np.zeros(nboot+1), 'm_cal_num_i': np.zeros(nboot+1),
			 'stat_error_num_i': np.zeros(nboot+1), 'N_i': 0}
		return x

	m = dict_per_bin['m'][mask]
	weight = dict_per_bin['weight'][mask]
	et = dict_per_bin['et'][mask]
	ex = dict_per_bin['ex'][mask]
	sigma_critic = dict_per_bin['sigma_critic'][mask]
	# Create boot sample
	if nboot>0:
		krand = np.random.choice(np.arange(N_i), (N_i, nboot), replace=True)
		m = np.hstack((m[:, np.newaxis], m[krand]))
		weight = np.hstack((weight[:, np.newaxis], weight[krand]))
		et = np.hstack((et[:, np.newaxis], et[krand]))
		ex = np.hstack((ex[:, np.newaxis], ex[krand]))
		sigma_critic = np.hstack((sigma_critic[:, np.newaxis], sigma_critic[krand]))
	
	w = weight/sigma_critic**2
	x = {}
	x['accum_w_i'] = np.sum(w, axis=0)
	x['m_cal_num_i'] = np.sum(w*(1+m), axis=0)
	x['shear_i'] = np.sum(w*et*sigma_critic, axis=0)
	x['cero_i'] = np.sum(w*ex*sigma_critic, axis=0)
	x['stat_error_num_i'] = np.sum( (w*0.25*sigma_critic)**2, axis=0)
	if nboot==0:
		for k in x.keys():
			if isinstance(x[k], np.ndarray): break
			x[k] = np.array([x[k]])

	x['N_i'] = N_i
	return x


def _profile_per_lens(j, dict_per_lens):
	data_L = dict_per_lens['data_L']
	data_S = dict_per_lens['data_S']
	bins = dict_per_lens['bins']
	back_dz = dict_per_lens['back_dz']
	nboot = dict_per_lens['nboot']
	#cosmo = dict_per_lens['cosmo']

	dL = data_L.iloc[j]
	dS = data_S.loc[dL['CATID']]
	#if back_dz != 0.:
	mask_dz = dS['Z_B'].values >= dL['Z'] + back_dz
	dS = dS[mask_dz]

	DD = gentools.compute_lensing_distances(zl=dL['Z'], zs=dS['Z_B'].values,
		dist=['DL', 'DS', 'DLS'], precomputed=True, cache=True)#, cosmo=self.cosmo)

	Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
	sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])
	
	# Compute distance and ellipticity components...
	dist, theta = gentools.sphere_angular_vector(dS['RAJ2000'].values, dS['DECJ2000'].values,
												dL['RA'], dL['DEC'], units='deg')
	theta += 90. 
	dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h
	et, ex = gentools.polar_rotation(dS['e1'].values, dS['e2'].values, np.deg2rad(theta))

	digit = np.digitize(dist_hMpc, bins=bins)-1
		
	dict_per_bin = {'m': dS['m'].values, 
					'weight': dS['weight'].values,
					'digit': digit, 
					'sigma_critic': sigma_critic,
					'et': et, 'ex': ex, 
					'nboot': nboot}

	xshape = (nboot+1, len(bins)-1)
	x = {'shear_j': np.zeros(xshape), 'cero_j': np.zeros(xshape),
		 'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
		 'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(len(bins)-1)}

	for i in range(len(bins)-1):
		pf = _profile_per_bin(i, dict_per_bin)
		x['shear_j'][:, i] = pf['shear_i']
		x['cero_j'][:, i] = pf['cero_i']
		x['accum_w_j'][:, i] = pf['accum_w_i']
		x['m_cal_num_j'][:, i] = pf['m_cal_num_i']
		x['stat_error_num_j'][:, i] = pf['stat_error_num_i']
		x['N_j'][i] = pf['N_i']

	return x
# =============================================================================================

def read_profile(self, file, colnames=True):
	''' Read profile written with Profile.write_to() method
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

class Profile(object):

	def __init__(self, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log',
		nboot=0, cosmo=cosmo, back_dz=0., precompute_distances=True, njobs=1):
		
		#if not isinstance(cat, ExpandedCatalog):
		#	raise TypeError('cat must be a LensCat.ExpandedCatalog catalog.')

		self.cosmo = cosmo
		self.nboot = nboot
		self.njobs = njobs
		self.back_dz = back_dz

		self.bins = gentools.make_bins(rin=rin_hMpc, rout=rout_hMpc, bins=bins, space=space)
		self.nbin = len(self.bins)-1
		self.rin_hMpc  = self.bins[0]
		self.rout_hMpc = self.bins[-1]
		self.r_hMpc = 0.5 * (self.bins[:-1] + self.bins[1:])	


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

@gentools.timer
class ExpandedProfile(Profile):

	def __init__(self, data=None, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log', 
		nboot=0, cosmo=cosmo, back_dz=0., precompute_distances=True, njobs=1):
		
		super().__init__(rin_hMpc, rout_hMpc, bins, space, 
							nboot, cosmo, back_dz, precompute_distances, njobs)

		pf = self._profile(data=data)
		pf = self._reduce(pf)
		
		# Now in units of h*Msun/pc**2
		self.shear = pf['shear']/self.cosmo.h
		self.cero = pf['cero']/self.cosmo.h
		self.shear_error = pf['shear_error']/self.cosmo.h
		self.cero_error = pf['cero_error']/self.cosmo.h
		self.stat_error = pf['stat_error']/self.cosmo.h
		self.N = pf['N'].astype(np.int32)

	def _reduce(self, pf):

		shear = np.array( [pf[_]['shear_i'] for _ in range(len(pf))] ).T
		cero = np.array( [pf[_]['cero_i'] for _ in range(len(pf))] ).T
		accum_w = np.array( [pf[_]['accum_w_i'] for _ in range(len(pf))] ).T
		m_cal_num = np.array( [pf[_]['m_cal_num_i'] for _ in range(len(pf))] ).T
		stat_error_num = np.array( [pf[_]['stat_error_num_i'] for _ in range(len(pf))] ).T
		N = np.array( [pf[_]['N_i'] for _ in range(len(pf))] )

		m_cal = m_cal_num / accum_w
		shear = (shear/accum_w) / m_cal
		cero = (cero/accum_w) / m_cal
		stat_error = np.sqrt(stat_error_num/accum_w**2) / m_cal

		x = {}
		x['shear'] = shear[0, :]
		x['cero'] = cero[0, :]
		x['stat_error'] = stat_error[0, :]
		x['shear_error'] = np.std(shear[1:,:], axis=0)
		x['cero_error'] = np.std(cero[1:,:], axis=0)
		x['N'] = N.astype(np.int32)

		return x

	def _profile(self, data):
		''' Computes profile for ExpandedCatalog
		'''
		#if self.back_dz != 0.:
		mask_dz = data['Z_B'] >= data['Z'] + self.back_dz
		data = data[mask_dz]

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
			
		dict_per_bin = {'m': data['m'].values,
		 				'weight': data['weight'].values,
						'digit': digit, 
						'sigma_critic': sigma_critic.values,
						'et': et.values, 'ex': ex.values,
						'nboot': self.nboot}

		with Parallel(n_jobs=self.njobs) as parallel:
			delayed_fun = delayed(_profile_per_bin)
			pf = parallel(delayed_fun(i, dict_per_bin) for i in range(self.nbin))

		return pf


@gentools.timer
class CompressedProfile(Profile):

	def __init__(self, data_L, data_S, rin_hMpc=0.1, rout_hMpc=10., bins=10, space='log',
		nboot=0, cosmo=cosmo, back_dz=0., precompute_distances=True, njobs=1):
		
		#if not isinstance(cat, (CompressedCatalog, ExpandedCatalog)):
		#	raise TypeError('cat must be a LensCat catalog.')

		super().__init__(rin_hMpc, rout_hMpc, bins, space, 
							nboot, cosmo, back_dz, precompute_distances, njobs)

		if data_S.index.name is not 'CATID':
			data_S_indexed = data_S.set_index('CATID')
		pf = self._profile(data_L=data_L, data_S=data_S_indexed)
		pf = self._reduce(pf)

		# Now in units of h*Msun/pc**2
		self.shear = pf['shear']/self.cosmo.h
		self.cero = pf['cero']/self.cosmo.h
		self.shear_error = pf['shear_error']/self.cosmo.h
		self.cero_error = pf['cero_error']/self.cosmo.h
		self.stat_error = pf['stat_error']/self.cosmo.h
		self.N = pf['N'].astype(np.int32)


	def _reduce(self, pf):

		shear = np.sum( [pf[_]['shear_j'] for _ in range(len(pf))], axis=0)
		cero = np.sum( [pf[_]['cero_j'] for _ in range(len(pf))], axis=0)
		accum_w = np.sum( [pf[_]['accum_w_j'] for _ in range(len(pf))], axis=0)
		m_cal_num = np.sum( [pf[_]['m_cal_num_j'] for _ in range(len(pf))], axis=0)
		stat_error_num = np.sum( [pf[_]['stat_error_num_j'] for _ in range(len(pf))], axis=0)
		N = np.sum( [pf[_]['N_j'] for _ in range(len(pf))], axis=0)

		m_cal = m_cal_num / accum_w
		shear = (shear/accum_w) / m_cal
		cero = (cero/accum_w) / m_cal
		stat_error = np.sqrt(stat_error_num/accum_w**2) / m_cal

		x = {}
		x['shear'] = shear[0, :]
		x['cero'] = cero[0, :]
		x['stat_error'] = stat_error[0, :]
		x['shear_error'] = np.std(shear[1:,:], axis=0)
		x['cero_error'] = np.std(cero[1:,:], axis=0)
		x['N'] = N	
		return x

	def _profile(self, data_L, data_S):
		''' Computes profile for CompressedCatalog
		'''
		dict_per_lens = {'data_L': data_L, 'data_S': data_S, 
						'bins': self.bins, 'back_dz': self.back_dz, 'nboot': self.nboot}

		# Compute profiles per lens
		with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
			delayed_fun = delayed(_profile_per_lens)
			pf = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
		return pf
