import os
import numpy as np
import pandas as pd
from astropy.io import fits
import h5py
from main import Survey, classonly, read_columns, cat_paths


class CS82(Survey):
	
	name = 'CS82'
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
			fields = ['CS1']
		if columns is None:
			columns = ['RAJ2000','DECJ2000','Z_B','e1','e2','m','weight','ODDS','fitclass','MASK']

		# Somehow we load the data and save it in cls.data
		field_paths = [cat_paths.cs82[field] for field in fields]
		dl = pd.concat( [read_columns(path, columns) for path in field_paths] ).reset_index(drop=True)
		catid = cls.name+'.'+pd.DataFrame({'CATID': np.arange(dl.shape[0]).astype(str)})
		cls.data = pd.concat([catid,dl], axis=1)

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