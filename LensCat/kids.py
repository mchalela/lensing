import os
import numpy as np
import pandas as pd
from astropy.io import fits
import h5py

from ..gentools import classonly
from .main import Survey, read_columns, cat_paths

class KiDS(Survey):
	
	name = 'KiDS'
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
			fields = ['G9','G12','G15','G23','GS']
		if columns is None:
			columns = ['RAJ2000','DECJ2000','Z_B','e1','e2','m','weight','ODDS','fitclass','MASK']

		# Somehow we load the data and save it in cls.data
		field_paths = [cat_paths.kids[field] for field in fields]
		dl = pd.concat( [read_columns(path, columns) for path in field_paths] ).reset_index(drop=True)
		catid = 2*10**8 + np.arange(dl.shape[0]).astype(np.int32)
		catid = pd.DataFrame({'CATID': catid})
		cls.data = pd.concat([catid,dl], axis=1)

		# Science cuts...
		# fitclass=0 for galaxies
		# MASK<=1 for objects outside the masked regions
		# weight>0 to have non-negative weights
		if science_cut:
			mask = (cls.data['fitclass']==0) & (cls.data['MASK']<=1) & (cls.data['weight']>0)
			mask &= cls.data['ODDS']>=0.4
			cls.data = cls.data[mask]
			cls.data.drop(columns=['fitclass', 'MASK'], inplace=True)

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
