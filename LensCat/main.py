import os, platform
import numpy as np
import pandas as pd
import h5py
import itertools
from astropy.io import fits
from astropy.table import Table

from grispy import GriSPy
from ..gentools import classonly
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

class cat_paths:
	'''Define your local paths
	'''
	node = platform.node()
	if node in ['mirta2','mirta3','sersic','clemente', \
				'clemente01','clemente02','clemente03','clemente04']:
		p = '/mnt/is0/mchalela/lensing/'
	elif node in ['univac','multivac']:
		p = '/home/martin/Documentos/Doctorado/Lentes/lensing/'
	else:
		raise ValueError, 'There is no catalog path for the node: '+node

	cs82 = {'CS1': os.path.join(p,'CS82','cs82_combined_lensfit_sources_nozcuts_aug_2015.h5')}

	kids = {'G9': os.path.join(p,'KiDS','KiDS_DR3.1_G9_ugri_shear.h5'),
			'G12': os.path.join(p,'KiDS','KiDS_DR3.1_G12_ugri_shear.h5'),
			'G15': os.path.join(p,'KiDS','KiDS_DR3.1_G15_ugri_shear.h5'),
			'G23': os.path.join(p,'KiDS','KiDS_DR3.1_G23_ugri_shear.h5'),
			'GS': os.path.join(p,'KiDS','KiDS_DR3.1_GS_ugri_shear.h5')}
	
	cfht = {'W1': os.path.join(p,'CFHT','CFHTLens_W1.h5'),
			'W2': os.path.join(p,'CFHT','CFHTLens_W2.h5'),
			'W3': os.path.join(p,'CFHT','CFHTLens_W3.h5'),
			'W4': os.path.join(p,'CFHT','CFHTLens_W4.h5')}

	rcsl = {'RCSL1': os.path.join(p,'RCSL','RCSLens.h5')}


def read_columns(hdf5file, columns):
	'''Read catalogue columns into pandas DataFrame
	'''
	file = h5py.File(hdf5file)
	groups = file.keys()

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
	gsp = GriSPy(pos, N_cells=N_cells, periodic=periodic, metric='sphere')
	return gsp

class formats:
	'''Convert dtypes between diferent formats
	'''
	np2fits = {np.int64:'1K', np.int32:'1J', np.int16:'1I',
			   np.float64:'1D', np.float32:'1E', np.bool:'L'}
	np2fits_obj = {str:'20A', list:'PJ()'}

	@classonly
	def pd2fits(cls, column):
		'''Pandas uses numpy as base data type
		'''
		is_dtype_equal = pd.core.dtypes.common.is_dtype_equal
		dtype = column.dtype
		inner_dtype = type(column.iloc[0])
		# este if es muy pero muy horrible...
		if dtype in list(cls.np2fits.keys()):
			for npfmt in cls.np2fits.keys():
				if is_dtype_equal(dtype, npfmt):
					return cls.np2fits[npfmt]
		elif inner_dtype in list(cls.np2fits_obj.keys()):
			for npfmt in cls.np2fits_obj.keys():
				if is_dtype_equal(inner_dtype, npfmt):
					return cls.np2fits_obj[npfmt]
	
	@classonly
	def pd2hdf5(cls, dtype):
		return None
#
###############################################################################

class Catalog(object):
	'''Class for catalog of source galaxies
	'''

	def __init__(self, data=None, name='Catalog', cat_type='expanded'):
		self._data = data
		self._LensID = 'ID'
		self.name = name
		if data is None:
			self.sources = 0
		else:
			self.sources = data.shape[0]
		
		assert cat_type.lower() in ['expanded', 'lenses', 'sources'], \
			'cat_type not recognized. Use: Expanded, Lenses or Sources.'
		self.cat_type = cat_type.lower()

	def __add__(self, catalog2):
		assert self.cat_type == 'lenses', \
			'Operator + is only defined for lenses catalog. You sended: {}'.format(self.cat_type)

		if self.data.index.name == self.LensID:
			keepID=True
			self.data.reset_index(drop=False, inplace=True)
		else:
			keepID=False

		catalog3 = Catalog(name=self.name+'+'+catalog2.name, cat_type='lenses')
		catalog3.data = pd.concat([self.data, catalog2.data])
		catalog3.data = catalog3.data.drop_duplicates(subset=self.LensID, keep='first')
		if keepID: catalog3.data.set_index(self.LensID, inplace=True)
		return catalog3

	def __and__(self, catalog2):
		catalog3 = Catalog(name=self.name+'&'+catalog2.name)
		catalog3.data = pd.concat([self.data, catalog2.data]).reset_index(drop=True)
		return catalog3

	def __str__(self):
		output = '{0} Catalog\n'.format(self.cat_type.capitalize())+ \
				'Name: {0}\n'.format(self.name)
		if self.cat_type in	['expanded', 'sources']:
			output += 'Sources: {0}\n'.format(self.sources)
		else:
			output += 'Lenses: {0}\n'.format(self.sources)
		if self.cat_type != 'sources':
			output += 'LensID: {0}\n'.format(self.LensID)
		output += 'Columns: {0}'.format(list(self.data.columns.values))
		return output

	def __repr__(self):
		return self.__str__()

	@property
	def LensID(self):
		return self._LensID
	@LensID.setter
	def LensID(self, LensID):
		if LensID in list(self.data.columns.values):
			self._LensID = LensID
		else:
			raise KeyError, '{} is not a valid column name.'.format(LensID)

	@property
	def data(self):
		return self._data
	@data.setter
	def data(self, data):
		if data is not None:
			self._data = data
			self.sources = data.shape[0]

	def add_column(self, index=0, name=None, data=None):
		'''Add a column in a given index position
		'''
		self.data.insert(index, name, data)

	def remove_column(self, columns):
		'''Remove a list of columns
		'''
		self.data.drop(columns=columns)

	def write_to(self, file, format='FITS', overwrite=False):
		'''Save the source catalog
		'''
		if format=='FITS':
			cols = []
			for name in list(self.data.columns):
				c = fits.Column(name=str(name), format=formats.pd2fits(self.data[name]), array=self.data[name].to_numpy())
				cols.append(c)
			hdul = fits.BinTableHDU.from_columns(cols)
			for line in hdul.header.items():
				if 'TFORM' in line[0]:
					del hdul.header[line[0]]
					hdul.header.append(line, end=True)
			hdul.header.insert(9, ('CATNAME', self.name))
			hdul.header.insert(10, ('CATTYPE', self.cat_type))
			hdul.writeto(file, overwrite=overwrite)
		else:
			print 'Format '+format+' is not implemented yet. Use "FITS"'
		pass

	@classonly
	def read_catalog(self, file):
		'''Load catalog saved with the Catalog.write_to() method
		'''
		cat = Catalog()
		with fits.open(file) as f:
			cat.data = Table(f[1].data).to_pandas()
			cat.name = f[1].header['CATNAME']
			#cat.cat_type = f[1].header['CATTYPE']
			cat.sources = f[1].header['NAXIS2']		
		return cat


class Survey(object):
	'''Class to handle all lensing catalogs
	'''

	name = 'Survey'
	periodic = {0:(0,360)}

	def __init__(self):
		self.region = None
		self.data = None

	@classonly
	def find_neighbors(cls, centre, upper_radii, lower_radii=None, append_data=None, compressed=False, njobs=4):

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
			dd, ii = cls.gsp.bubble_neighbors(centre, distance_upper_bound=upper_radii, njobs=njobs)
		else: 
			dd, ii = cls.gsp.shell_neighbors(centre, distance_upper_bound=upper_radii, 
						distance_lower_bound=lower_radii, njobs=njobs)

		if cls.compressed:
			# Two catalogs, one for galaxies and one for groups
			lens_cat = Catalog(name=cls.name, cat_type='lenses')
			src_per_lens = np.array( map(len, dd) )
			mask_nsrc = src_per_lens>0
			cat_ids = [list(cls.data['CATID'].iloc[_]) for _ in ii]
			dic ={'CATNAME': np.tile([cls.name], len(ii)),
				'N_SOURCES': src_per_lens, 
				'CATID': np.array(cat_ids, dtype=np.object) }
			ii_data = pd.DataFrame(dic)
			if append_data is not None:
				append_data.reset_index(drop=True, inplace=True)
				lens_cat.data = pd.concat([append_data, ii_data], axis=1)[mask_nsrc]
			else:
				lens_cat.data = ii_data[mask_nsrc].reset_index(drop=True)

			src_cat = Catalog(name=cls.name, cat_type='sources')
			ii = list(itertools.chain.from_iterable(ii))
			iu, im = np.unique(ii, return_counts=True)
			extra_data = pd.DataFrame({'CATNAME': np.tile([cls.name], len(iu)),
				'MULTIPLICITY': im})
			iu_data = cls.data.iloc[iu].reset_index(drop=True)
			src_cat.data = pd.concat([iu_data, extra_data], axis=1).reset_index(drop=True)
			#src_cat.data = cls.data.iloc[iu].reset_index(drop=True)
			#src_cat.data['CATID'] = cls.name+'.'+src_cat.data['CATID'].astype(str)
			return lens_cat, src_cat

		else:
			# One catalog with repeated galaxies	
			cat = Catalog(name=cls.name, cat_type='expanded')
			ii = list(itertools.chain.from_iterable(ii))
			cat.data = cls.data.iloc[ii].reset_index(drop=True)

			# Append lens data to sources catalog
			if append_data is not None:
				src_per_lens = np.array( map(len,dd) )
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



