import os
import platform

import numpy as np 

from astropy.io import fits


# =============================================================================
# Define local paths

class cat_paths:
    '''Define your local paths
    '''
    node = platform.node()
    if node in ['mirta2','mirta3','sersic','clemente', \
                'clemente01','clemente02','clemente03','clemente04']:
        p = '/home/mchalela/lensing/'
    elif node in ['univac','multivac']:
        p = '/home/martin/Documentos/Doctorado/Lentes/lensing/'
    elif node in ['jaffa']:
        p = '/home/martin/Documents/Lentes/lensing/'
    else:
        raise ValueError('There is no catalog path for the node: '+node)

    # Catalogs paths
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

    # Catalog masks path
    field_mask = {'KiDS': os.path.join(p,'KiDS','KiDS_DR3.1_mask.fits'),
                'CFHT': os.path.join(p,'CFHT','CFHTLens_mask.fits'),
                'RCSL': os.path.join(p,'RCSL','RCSLens_mask.fits'),
                'CS82': os.path.join(p,'CS82','cs82_aug_2015_mask.fits')}


# =============================================================================
# Compute field masks

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

def sky2pix(ra, dec, pix_size):
    ''' Convert ra,dec to x,y given a pix_size in degrees
    '''
    ra_nbins, dec_nbins = int(360/pix_size) + 1, int(180/pix_size) + 1
    
    # Create bins so that pixel (0, 0) corresponds to sky (0, 90)
    ra_bins = np.linspace(0., 360., ra_nbins)
    dec_bins = np.linspace(90., -90., dec_nbins)

    ra_digit = digitize(ra, ra_bins)
    dec_digit = digitize(dec, dec_bins)

    return ra_digit, dec_digit

def build_mask(ra, dec, pix_size=0.1, catname='CAT'):
    '''Builds boolean mask of the full sky. This will create 
    a pixel grid in RA from 0 to 360, and DEC from -90 to 90,
    and return an ImageHDU.

    ra: numpy array of RA
    dec: numpy array of DEC
    pix_size: float
        This value is the step in degrees for the RA-DEC grid.
        It will be saved in the header as PIX_SIZE
    catname: string
        Name of the catalog.
        It will be saved in the header as CATALOG
    '''

    ra_nbins, dec_nbins = int(360/pix_size) + 1, int(180/pix_size) + 1
    
    ra_digit, dec_digit = sky2pix(ra, dec, pix_size)

    mask = np.zeros((dec_nbins, ra_nbins), dtype=bool)
    mask[dec_digit, ra_digit] = True

    cards = [('CATALOG', catname), ('PIX_SIZE', pix_size)]
    img = array2hdul(mask.astype(np.uint8), new_cards=cards)
    
    return img


def infield(ra, dec, catname):
    '''Check if given positions RA, DEC lie within the region of CATNAME
    '''
    assert catname in ('CS82', 'KiDS', 'RCSL', 'CFHT'), \
        f'catname = {catname} is not a valid catalog name.'

    with fits.open(cat_paths.field_mask[catname]) as fm:
        data = fm[1].data
        xpix, ypix = sky2pix(ra, dec, pix_size=0.1)
        mask = data[ypix, xpix].astype(np.bool) #.flatten()

    return mask
