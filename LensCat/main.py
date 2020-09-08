import os
import platform

import numpy as np
import pandas as pd
import h5py

import itertools

from astropy.io import fits
from astropy.table import Table

from grispy import GriSPy

from ..gentools import classonly
from .fields import infield, cat_paths
'''
Que tipo de comportamiento espero?

Algo asi: 
import lenscat
kids_cat = lenscat.KiDS.bubble_neighbors(alpha, delta, radio)	# pero sin instanciar ninguna clase. Son classmethod?

Quiero que kids_cat sea una estructura similar al hdulist
que se pueda acceder a los datos con nombre:
kids_cat['E1'], kids_cat['E2']

Propiedades: con @property puede ser
  Que tenga un header (de astropy) y un arreglo con todos los datos:
  kids_cat.header, kids_cat.data
  Que tenga notas o comentarios:
  kids_cat.notes		# cosas como magnitud limite o cobertura en grados cuadrados...

Metodos:
  Que se pueda escribir el catalogo:
  kids_cat.write_to()
  Despues leerlo
  kids_cat = lenscat.read_catalog()

  Quitar y agregar columnas:
  kids_cat.add_column(), kids_cat.remove_column()

  Chequear si objetos caen en el campo del catalogo
  lenscat.KiDS.in_field(alpha, delta)
  o en alguna mascara:
  lenscat.KiDS.in_mask(alpha, delta)

  Deberia haber un metodo que cargue el catalogo y otro que lo libere
  lenscat.KiDS.load(), lenscat.KiDS.drop()
'''

#------------------------------------------------------------------------------

def read_columns(hdf5file, columns):
	'''Read catalogue columns into pandas DataFrame
	'''
	file = h5py.File(hdf5file, 'r')
	groups = list(file.keys())

	data = pd.DataFrame()
	for i, col in enumerate(columns):
		for grp in groups:
			dset = file['{0}/objects'.format(grp)]
			fields = list(dset.dtype.fields)
			if col in fields:
				data.insert(i, col, dset[col])
				break
	return data.reset_index(drop=True)

def call_gripsy(data):
	'''Creates a GriSPy grid
	'''	
	N_cells = 50
	periodic = {0: (0,360)}
	pos = data[['RAJ2000', 'DECJ2000']].to_numpy()
	gsp = GriSPy(pos, N_cells=N_cells, periodic=periodic, metric='vincenty')
	return gsp

class formats:
	'''Convert dtypes between diferent formats.
	'''
	np2fits = {np.int64:'1K', np.int32:'1J', np.int16:'1I',
			   np.float64:'1D', np.float32:'1E', np.bool:'L'}
	np2fits_obj = {str:'20A', list:'QJ()', np.ndarray:'QJ()'}

	@classonly
	def pd2fits(cls, column):
		'''Pandas uses numpy as base data type
		'''
		is_dtype_equal = pd.core.dtypes.common.is_dtype_equal
		dtype = column.dtype
		inner_dtype = type(column.iloc[0])
		# este if es muy pero muy horrible...
		if dtype in list(cls.np2fits.keys()):
			for npfmt in list(cls.np2fits.keys()):
				if is_dtype_equal(dtype, npfmt):
					return cls.np2fits[npfmt]
		elif inner_dtype in list(cls.np2fits_obj.keys()):
			for npfmt in list(cls.np2fits_obj.keys()):
				if is_dtype_equal(inner_dtype, npfmt):
					return cls.np2fits_obj[npfmt]
	
	@classonly
	def pd2hdf5(cls, dtype):
		return None

###############################################################################

def read_catalog(file):
	'''Load catalog saved with the Catalog.write_to() method
	'''
	with fits.open(file) as f:
		cat_type = f[1].header['CATTYPE']
		name = f[1].header['CATNAME']
		LensID = f[1].header['LENSID']
		if LensID == 'None': LensID=None
		if cat_type.lower() == 'expanded':
			data = Table(f[1].data).to_pandas()
			cat = ExpandedCatalog(name=name, data=data, LensID=LensID)
		elif cat_type.lower() == 'compressed':
			data_L = Table(f[1].data).to_pandas()
			data_S = Table(f[2].data).to_pandas()
			cat = CompressedCatalog(name=name, data_L=data_L, data_S=data_S, LensID=LensID)
		else:
			raise ValueError('CATTYPE = {} is not a valid catalog.'.format(cat_type))
	return cat



class ExpandedCatalog(object):

	def __init__(self, name='Catalog', data=None, LensID=None):
		self.cat_type = 'Expanded'
		self.name = name
		self._LensID = LensID
		self._data = data
		self.sources = 0 if data is None else data.shape[0]
		if (LensID is not None) and (data is not None):
			self.lenses = data[LensID].unique().shape[0]
		else:
			self.lenses = 0

	def __str__(self):
		output = '<Expanded Catalog>\n'
		output += 'Name: {0}\n'.format(self.name)
		output += 'LensID: {0}\n'.format(self.LensID)
		output += 'Lenses: {0}\n'.format(self.lenses)
		output += 'Sources: {0}\n'.format(self.sources)
		output += 'Columns: {0}\n'.format(list(self.data.columns))
		return output

	def __repr__(self):
		return self.__str__()

	def __add__(self, catalog2):
		raise AttributeError('Addition not implemented for Expanded Catalogs. Use "&".')
		return None

	def __and__(self, catalog2):
		catalog3 = Catalog(name=self.name+'&'+catalog2.name)
		catalog3.data = pd.concat([self.data, catalog2.data]).reset_index(drop=True)
		return catalog3

	@property
	def LensID(self):
		return self._LensID

	@LensID.setter
	def LensID(self, LensID):
		colnames = [self.data.index.name] + list(self.data.columns)
		assert LensID in colnames, '{} is not a valid column name.'.format(LensID)
		self._LensID = LensID
		self.lenses = self.data[LensID].unique().shape[0]

	@property
	def data(self):
		return self._data
	@data.setter
	def data(self, newdata):
		if newdata is not None:
			self._data = newdata
			self.sources = newdata.shape[0]
			self.lenses = self.data[self.LensID].unique().shape[0]


	def add_column(self, index=0, name=None, data=None):
		'''Add a column in a given index position
		'''
		assert self.data.shape[0] == data.shape[0], \
				'Number of sources ({}) does not match \
				the number of rows in data ({})'.format(self.data.shape[0], data.shape[0])
		self.data.insert(index, name, data)

	def remove_column(self, columns, inplace=False):
		'''Remove a list of columns
		'''
		self.data.drop(columns=columns, inplace=inplace)

	def write_to(self, file, format='FITS', overwrite=False):
		'''Save the source catalog
		'''
		assert format.upper()=='FITS', 'Format '+format+' is not implemented yet. Use "FITS"'
		
		cols = []
		for name in list(self.data.columns):
			c = fits.Column(name=str(name), 
				format=formats.pd2fits(self.data[name]), array=self.data[name].to_numpy())
			cols.append(c)
		hdu = fits.BinTableHDU.from_columns(cols)
		for line in list(hdu.header.items()):
			if 'TFORM' in line[0]:
				del hdu.header[line[0]]
				hdu.header.append(line, end=True)
		hdu.header.insert(8, ('CATNAME', self.name))
		hdu.header.insert(9, ('CATTYPE', 'EXPANDED'))
		hdu.header.insert(10, ('LENSID', str(self.LensID)))
		hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
		hdulist.writeto(file, overwrite=overwrite)


class CompressedCatalog(object):

	def __init__(self, name='Catalog', data_L=None, data_S=None, LensID=None):
		self.cat_type = 'Compressed'
		self.name = name
		self._LensID = LensID
		self._data_L = data_L
		self._data_S = data_S
		self.lenses = 0 if data_L is None else data_L.shape[0]
		self.sources = 0 if data_S is None else data_S.shape[0]

	def __str__(self):
		output = '<Compressed Catalog>\n'
		output += 'Name: {0}\n'.format(self.name)
		output += 'LensID: {0}\n'.format(self.LensID)
		output += 'Lenses: {0}\n'.format(self.lenses)
		output += 'Sources: {0}\n'.format(self.sources)
		output += 'Lenses Columns: {0}\n'.format(list(self.data_L.columns))
		output += 'Sources Columns: {0}\n'.format(list(self.data_S.columns))
		return output

	def __repr__(self):
		return self.__str__()

	def __or__(self, catalog2):
		''' Concatenate catalogs keeping the lens from the left catalog in the case
		it is also present in the right catalog.
		'''
		assert self.LensID is not None, \
			'Please set the LensID property with a valid column name from data_L.'

		keepID_L, keepID_S = False, False
		if self.data_L.index.name == self.LensID:
			keepID = True
			self.data_L.reset_index(drop=False, inplace=True)
		if self.data_S.index.name == 'CATID':
			keepID_S = True
			self.data_S.reset_index(drop=False, inplace=True)

		# Add lenses data
		catalog3 = CompressedCatalog(name=self.name+'|'+catalog2.name, LensID=self.LensID)
		catalog3.data_L = pd.concat([self.data_L, catalog2.data_L])
		catalog3.data_L = catalog3.data_L.drop_duplicates(subset=self.LensID, keep='first')
		# Add sources data
		catalog3.data_S = pd.concat([self.data_S, catalog2.data_S])
		# Restore IDs
		if keepID_L: catalog3.data_L.set_index(self.LensID, inplace=True)
		if keepID_S: catalog3.data_S.set_index('CATID', inplace=True)
		return catalog3


	def __add__(self, catalog2):
		''' Concatenate catalogs with no Survey superposition. If a lens is present
		in both catalogs, this method will identify the superposition in precomputed 
		masks. If there is superposition in a region of the sky, the galaxies in 
		the catalog of the left will be prioritized.
		'''
		assert (self.LensID is not None) and (catalog2.LensID is not None), \
			'Please set the LensID property with a valid column name from data_L.'

		idx_is_LensID = self.data_L.index.name == self.LensID
		idx_is_CATID = self.data_S.index.name == 'CATID'

		if not idx_is_LensID:
			self.data_L.set_index(self.LensID, drop=True, inplace=True)
			catalog2.data_L.set_index(catalog2.LensID, drop=True, inplace=True)

		if not idx_is_CATID:
			self.data_S.set_index('CATID', drop=True, inplace=True)
			catalog2.data_S.set_index('CATID', drop=True, inplace=True)

		# ------------------------------------------------
		catnames = self.name.split('+')

		# compute superposition mask
		pos = catalog2.data_S[['RAJ2000', 'DECJ2000']]
		superposition_mask = np.zeros(len(pos), dtype=bool)
		for catname in catnames:
			superposition_mask |= infield(
									ra=pos.RAJ2000.values,
									dec=pos.DECJ2000.values,
									catname=catname)

		# delete from data_S
		catid2drop = catalog2.data_S.index[superposition_mask]
		catalog2.data_S.drop(catid2drop, inplace=True)

		# merge catalogs data_S
		catalog3 = CompressedCatalog(name=self.name+'+'+catalog2.name, LensID=self.LensID)
		catalog3.data_S = pd.concat([self.data_S.reset_index(drop=False),
									catalog2.data_S.reset_index(drop=False)])

		# merge catalogs data_L
		catalog3.data_L = pd.concat([self.data_L.reset_index(drop=False),
									catalog2.data_L.reset_index(drop=False)])
		catalog3.data_L.drop_duplicates(subset=self.LensID, keep='first', inplace=True)
		catalog3.data_L.set_index(self.LensID, drop=True, inplace=True)

		# add sources from repeated lenses
		repeated_idxs = self.data_L.index.intersection(catalog2.data_L.index)

		for idx in repeated_idxs:
			catid2 = catalog2.data_L['CATID'].loc[idx]
			catid3 = catalog3.data_L['CATID'].loc[idx]
			catalog3.data_L['CATID'].loc[idx] = np.hstack((catid3, np.setxor1d(catid2, catid2drop)))

		# fix N_SOURCES
		catalog3.data_L['N_SOURCES'] = np.fromiter(map(len, catalog3.data_L['CATID']), dtype=np.int32)

		# ------------------------------------------------
		# Restore IDs
		if not idx_is_LensID:
			self.data_L.reset_index(drop=False, inplace=True)
			catalog2.data_L.reset_index(drop=False, inplace=True)

		if not idx_is_CATID:
			self.data_S.reset_index(drop=False, inplace=True)
			catalog2.data_S.reset_index(drop=False, inplace=True)

		catalog3.data_L.reset_index(drop=False, inplace=True)
		catalog3.data_S.reset_index(drop=False, inplace=True)
		return catalog3

	def __and__(self, catalog2):
		''' Concatenate catalogs without filtering.
		This means that a single lens can have duplicated source
		galaxies from diferent surveys.
		'''

		keepID_L, keepID_S = False, False
		if self.data_L.index.name == self.LensID:
			keepID = True
			self.data_L.reset_index(drop=False, inplace=True)
		if self.data_S.index.name == 'CATID':
			keepID_S = True
			self.data_S.reset_index(drop=False, inplace=True)

		catalog3 = CompressedCatalog(name=self.name+'&'+catalog2.name)
		catalog3.data_L = pd.concat([self.data_L, catalog2.data_L])
		catalog3.data_S = pd.concat([self.data_S, catalog2.data_S])
		# Restore IDs
		if keepID_L: catalog3.data_L.set_index(self.LensID, inplace=True)
		if keepID_S: catalog3.data_S.set_index('CATID', inplace=True)		
		return catalog3

	@property
	def LensID(self):
		return self._LensID

	@LensID.setter
	def LensID(self, LensID):
		colnames = [self.data_L.index.name] + list(self.data_L.columns)
		assert LensID in colnames, '{} is not a valid column name.'.format(LensID)
		self._LensID = LensID

	@property
	def data_L(self):
		return self._data_L
	@data_L.setter
	def data_L(self, newdata_L):
		if newdata_L is not None:
			self._data_L = newdata_L
			self.lenses = newdata_L.shape[0]
	@property
	def data_S(self):
		return self._data_S
	@data_S.setter
	def data_S(self, newdata_S):
		if newdata_S is not None:
			self._data_S = newdata_S
			self.sources = newdata_S.shape[0]

	def add_column(self, index=0, name=None, data=None):
		'''Add a column in a given index position
		'''
		assert self.data_L.shape[0] == data.shape[0], \
				'Number of lenses ({}) does not match \
				the number of rows in data ({})'.format(self.data_L.shape[0], data.shape[0])
		self.data_L.insert(index, name, data)

	def remove_column(self, columns, inplace=False):
		'''Remove a list of columns
		'''
		self.data_L.drop(columns=columns, inplace=inplace)

	def write_to(self, file, format='FITS', overwrite=False):
		'''Save the source catalog
		'''
		assert format.upper()=='FITS', 'Format '+format+' is not implemented yet. Use "FITS"'
		
		hdulist = [fits.PrimaryHDU()]
		for data in [self.data_L, self.data_S]:
			cols = []
			for name in list(data.columns):
				c = fits.Column(name=str(name), 
					format=formats.pd2fits(data[name]), array=data[name].to_numpy())
				cols.append(c)
			hdu = fits.BinTableHDU.from_columns(cols)
			for line in list(hdu.header.items()):
				if 'TFORM' in line[0]:
					del hdu.header[line[0]]
					hdu.header.append(line, end=True)
			hdu.header.insert(8, ('CATNAME', self.name))
			hdu.header.insert(9, ('CATTYPE', 'COMPRESSED'))
			hdu.header.insert(10, ('LENSID', str(self.LensID)))
			hdulist.append(hdu)
		hdulist = fits.HDUList(hdulist)
		hdulist.writeto(file, overwrite=overwrite)




class Survey(object):
	'''Class to handle all lensing catalogs
	'''

	name = 'Survey'
	periodic = {0:(0,360)}

	def __init__(self):
		self.region = None
		self.data = None

	@classonly
	def find_neighbors(cls, centre, upper_radii, lower_radii=None, append_data=None, compressed=False, njobs=1):

		# Check if catalogue is loaded
		try:
			cls.data
		except AttributeError:
			cls.load()
		# Check if GriSPy was called
		try:
			cls.gsp
		except AttributeError:
			cls.gsp = call_gripsy(cls.data)

		cls.compressed = compressed
		# TODO: Check data types!! Create a class..
		if isinstance(centre, pd.DataFrame): centre = centre.to_numpy()

		# Search for sources
		if lower_radii is None:
			dd, ii = cls.gsp.bubble_neighbors(centre, distance_upper_bound=upper_radii)
		else: 
			dd, ii = cls.gsp.shell_neighbors(centre, distance_upper_bound=upper_radii, 
						distance_lower_bound=lower_radii)

		if cls.compressed:
			# One catalog, two data frames, one for galaxies and one for groups
			cat = CompressedCatalog(name=cls.name, LensID='ID')

			# Lenses data
			src_per_lens = np.fromiter(map(len, dd), dtype=np.int32)
			mask_nsrc = src_per_lens>0
			cat_ids = [list(cls.data['CATID'].iloc[_]) for _ in ii]
			dic ={'CATNAME': np.tile([cls.name], len(ii)),
				'N_SOURCES': src_per_lens, 
				'CATID': np.array(cat_ids, dtype=np.object) }
			ii_data = pd.DataFrame(dic)
			if append_data is not None:
				append_data.reset_index(drop=True, inplace=True)
				cat.data_L = pd.concat([append_data, ii_data], axis=1)[mask_nsrc]
			else:
				cat.data_L = ii_data[mask_nsrc].reset_index(drop=True)

			# Sources data
			ii = list(itertools.chain.from_iterable(ii))
			iu, im = np.unique(ii, return_counts=True)
			extra_data = pd.DataFrame({'CATNAME': np.tile([cls.name], len(iu)),
				'MULTIPLICITY': im})
			iu_data = cls.data.iloc[iu].reset_index(drop=True)
			cat.data_S = pd.concat([iu_data, extra_data], axis=1).reset_index(drop=True)
			# CHANGE Z_B PRECISION TO DOUBLE
			cat.data_S['Z_B'] = cat.data_S['Z_B'].astype(np.float64)
			return cat

		else:
			# One catalog with repeated galaxies	
			cat = ExpandedCatalog(name=cls.name, LensID='ID')
			ii = list(itertools.chain.from_iterable(ii))
			cat._data = cls.data.iloc[ii].reset_index(drop=True)
			# CHANGE Z_B PRECISION TO DOUBLE
			cat.data['Z_B'] = cat.data['Z_B'].astype(np.float64)

			# Append lens data to sources catalog
			if append_data is not None:
				src_per_lens = np.array( list(map(len, dd)) )
				append_data = append_data.loc[append_data.index.repeat(src_per_lens)]
				append_data.reset_index(drop=True, inplace=True)
				cat.data = pd.concat([cat.data, append_data], axis=1)
			return cat


	@classonly
	def in_field(cls, alpha, delta):
		return cls.region[alpha, delta]

	def add_column(cls, index=0, name=None, data=None):
		''' I'm not sure what I want this method to do...
		Either add a column provided by the user or simply load
		another column from the h5 files... maybe both?
		'''
		cls.data.insert(index, name, data)

	def remove_column(cls, columns):
		'''Remove a list of columns
		'''
		cls.data.drop(columns=columns)



