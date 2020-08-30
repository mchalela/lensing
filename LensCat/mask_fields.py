import numpy as np 
from astropy.io import fits


##############################################################################

def digitize(data, bins):
    """Return data bin index."""
    N = len(bins) - 1
    d = (N * (data - bins[0]) / (bins[-1] - bins[0])).astype(np.int)
    return d


def array2hdul(array, new_cards=[]):
	'''Return an ImageHDU with header
	array: numpy array
	new_cards: list of tuples with name and value
	'''
	img = fits.ImageHDU(array)
	if new_cards:
		for key, val in new_cards:
			img.header.append((key, val))
	return img


def build_mask(ra, dec, pix_size=0.1, catalog_name='CAT'):
	'''Builds boolean mask of the full sky.	This will create 
	a pixel grid in RA from 0 to 360, and DEC from -90 to 90,
	and return an ImageHDU.

	ra: numpy array of RA
	dec: numpy array of DEC
	pix_size: float
		This value is the step in degrees for the RA-DEC grid.
		It will be saved in the header as PIX_SIZE
	catalog_name: string
		Name of the catalog.
		It will be saved in the header as CATALOG
	'''

	ra_bins = np.arange(0., 360., pix_size)
	dec_bins = np.arange(-90., 90., pix_size)

	ra_digit = digitize(ra, ra_bins)
	dec_digit = digitize(dec, dec_bins)

	coords = np.array(list(zip(ra_digit, dec_digit)))

	mask = np.zeros((len(dec_bins), len(ra_bins)), dtype=bool)

	mask[coords[:,1], coords[:,0]] = True

	cards = [('CATALOG', catalog_name), ('PIX_SIZE', pix_size)]
	img = array2hdul(mask.astype(np.int), new_cards=cards)
	
	return img





