import os, platform
import numpy as np
import pandas as pd
import h5py
from grispy import GriSPy
import itertools
from astropy.io import fits
from astropy.table import Table

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
	if node in ['mirta2','mirta3','sersic','clemente']:
		p = '/mnt/is0/mchalela/lensing/'
	elif node in ['univac','multivac']:
		p = '/home/martin/Documentos/Doctorado/Lentes/lensing/'
	else:
		print 'There is no catalog path for the node: '+node

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
	np2fits = {np.object:'20A', np.int64:'1K', np.int32:'1J', np.int16:'1I',
			   np.float64:'1D', np.float32:'1E', np.bool:'L'}

	@classonly
	def pd2fits(cls, dtype):
		'''Pandas uses numpy as base data type
		'''
		is_dtype_equal = pd.core.dtypes.common.is_dtype_equal
		for npfmt in cls.np2fits.keys():
			if is_dtype_equal(dtype, npfmt):
				return cls.np2fits[npfmt]
	
	@classonly
	def pd2hdf5(cls, dtype):
		return None
#
###############################################################################

class Catalog(object):
	'''Class for catalog of source galaxies
	'''

	def __init__(self, data=None):
		self.data = data
		self.name = 'Catalog'
		self.sources = 0

	def __add__(self, catalog2):
		catalog3 = Catalog()
		catalog3.data = pd.concat([self.data, catalog2.data]).reset_index(drop=True)
		catalog3.name = self.name+'+'+catalog2.name
		return catalog3

	def __str__(self):
		output = 'Catalog: {0}\n'.format(self.name)+ \
				'Sources: {0}\n'.format(self.sources)+ \
				'Columns: {0}'.format(list(self.data.columns.values))
		return output

	def __repr__(self):
		return self.__str__()

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
				#print str(name), self.data[name].dtype, formats.pd2fits(self.data[name].dtype)
				c = fits.Column(name=str(name), format=formats.pd2fits(self.data[name].dtype), array=self.data[name].to_numpy())
				cols.append(c)
			hdul = fits.BinTableHDU.from_columns(cols)
			hdul.header.append(('CATNAME', self.name), end=True)
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
	def find_neighbors(cls, centre, upper_radii, lower_radii=0, append_data=None, njobs=4):

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

		# TODO: Check data types!! Create a class..
		if isinstance(centre, pd.DataFrame): centre = centre.to_numpy()

		# Search for sources
		if lower_radii==0:
			dd, ii = cls.gsp.bubble_neighbors(centre, distance_upper_bound=upper_radii, njobs=njobs)
		else: 
			dd, ii = cls.gsp.shell_neighbors(centre, distance_upper_bound=upper_radii, 
						distance_lower_bound=lower_radii, njobs=njobs)
		ii = list(itertools.chain.from_iterable(ii))
		cat = Catalog()
		cat.data = cls.data.iloc[ii].reset_index(drop=True)
		cat.name = cls.name

		# Append lens data to sources catalog
		if append_data is not None:
			sources_per_lens = np.array( map(len,dd) )
			append_data = append_data.loc[append_data.index.repeat(sources_per_lens)]
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



