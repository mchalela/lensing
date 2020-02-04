import os
import numpy as np
import pandas as pd
from astropy.io import fits
import h5py

from ..gentools import classonly
from .main import Survey, read_columns, cat_paths


class RCSL(Survey):
	
	name = 'RCSL'
	region = np.zeros(100, dtype=bool).reshape((10,10))
	region[6:,6:] = True
	
	def __init__(self):
		pass


	@classonly
	def load(cls, fields=None, columns=None, science_cut=True):
		'''Method to load the catalogue
		'''

		# Check some things...
		if fields is None:
			fields = ['RCSL1']
		if columns is None:
			columns = ['RAJ2000','DECJ2000','Z_B','e1','e2','m','weight','ODDS','fitclass','MASK',
						'c1_DP', 'c2_DP', 'c1_NB', 'c2_NB']

		# Somehow we load the data and save it in cls.data
		field_paths = [cat_paths.rcsl[field] for field in fields]
		dl = pd.concat( [read_columns(path, columns) for path in field_paths] ).reset_index(drop=True)
		catid = cls.name+'.'+pd.DataFrame({'CATID': np.arange(dl.shape[0]).astype(str)})
		cls.data = pd.concat([catid,dl], axis=1)

		# Additive correction, PONER REFERENCIAS DE COMO USAR LOS c1, c2
		#cls.data['e2'] -= cls.data['c2']
		#cls.data.drop(columns='c2', inplace=True)

		# Science cuts...
		# fitclass=0 for galaxies
		# MASK<=1 for objects outside the masked regions
		# weight>0 to have non-negative weights
		if science_cut:
			mask = (cls.data['fitclass']==0) & (cls.data['MASK']<=1) & (cls.data['weight']>0)
			cls.data = cls.data[mask]

	@classonly
	def drop(cls):
		'''Method to drop the Survey and GriSPy grid from memory
		'''
		try:
			del	cls.data
		except AttributeError:
			pass
		
		try:
			del cls.gsp
		except AttributeError:
			pass