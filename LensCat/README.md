# LensCat module description

## Catalogue Description

Some details and reference links for each catalogue...


## How to use it
Lets say you have your lens catalogue in a pandas data frame with an id, coordinates and some properties of interest: IDLENS, RA, DEC, z, prop1, prop2, prop3. It is important that the RA, DEC and z columns have exactly these names.
Now, you want all source galaxies in a radius of 1 Mpc. 

```python
import numpy as np
import pandas as pd
from lensing import gentools, LensCat
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)  # Standard cosmology

# Read your lens catalogue as a pandas DataFrame
cat = pd.read_csv('path/to/lens/catalogue.csv')

# We need the search radius in degrees, so we use Mpc2deg() from the gentools module.
R_Mpc = 1.
R_deg = gentools.Mpc2deg(R_Mpc=R_Mpc, z=cat['z'], cosmo=cosmo)

# Search for galaxies in the survey you want, and append your lens catalogue.
# You can lunch a parallel search in njobs.
cs82_cat = LensCat.CS82.bubble_neighbors(centre=cat[['RA','DEC']], radii=R_deg, append_data=cat, njobs=40)
kids_cat = LensCat.KiDS.bubble_neighbors(centre=cat[['RA','DEC']], radii=R_deg, append_data=cat, njobs=40)
cfht_cat = LensCat.CFHT.bubble_neighbors(centre=cat[['RA','DEC']], radii=R_deg, append_data=cat, njobs=40)
```

And thats it! You can save your catalogues individually or you can add them in a single catalogue.
```python
cs82_cat.write_to('gx_CS82.fits')
kids_cat.write_to('gx_KiDS.fits')
cfht_cat.write_to('gx_CFHT.fits')

# Or, you can add them and then save the file.
cat = cs82_cat + kids_cat + cfht_cat
cat.write_to('gx_'+cat.name+'.fits')
```

That was the easy way. By default, if you search for galaxies directly with the **bubble_neighbors()** method you will load only these columns from the catalogues: RAJ2000, DECJ2000, Z_B, e1, e2, m, weight, ODDS, fitclass, MASK. And in particular, the CFHTLens catalogue will also load the c2 additive bias column, correct the e2 component of ellipticity and drop the c2 column from the catalogue, thats why you wont see it.

This is because the catalogue needs to be loaded in memory first with the **load(fields=None, columns=None, science_cut=True)** method. If you don't load the catalogue before searching for galaxies, it will be loaded automatically with those default columns. You can control what columns and fields are loaded.
```python
# Lets say you want from the CS82 those columns and also the BPZ_LOW95
columns = ['RAJ2000','DECJ2000','Z_B','e1','e2','m','weight','ODDS','fitclass','MASK', 'BPZ_LOW95']
LensCat.CS82.load(columns=columns)
cs82_cat = LensCat.CS82.bubble_neighbors(centre=cat[['RA','DEC']], radii=R_deg, append_data=cat, njobs=40)

# Or lets say you want from the KiDS catalogue the default columns, but
# only in the G9 and G12 fields.
LensCat.KiDS.load(fields=['G9','G12'])
kids_cat = LensCat.KiDS.bubble_neighbors(centre=cat[['RA','DEC']], radii=R_deg, append_data=cat, njobs=40)
```

The science_cut will usually be True, and it gives you all the galaxies with **fitclass=0**, **MASK<=1** and **weight>0**. So if you want these cut you need to load these columns. For the CFHT in particular, the c2 column must be loaded.


If you want to save your catalogue to be used with the Shear or Kappa modules, you need to compute the angular diameter distances for the lensing analysis. For example:
```python
DL = np.array(cosmo.angular_diameter_distance(cs82_cat.data['z']))
DS = np.array(cosmo.angular_diameter_distance(cs82_cat.data['Z_B']))
DLS= np.array(cosmo.angular_diameter_distance_z1z2(cs82_cat.data['z'], cs82_cat.data['Z_B']))

# Save the columns after the IDLENS column. Omit the index if you dont care about the order
index = cs82_cat.data.columns.get_loc('IDLENS')
cs82_cat.add_column(index=index, name='DLS', data=DLS)
cs82_cat.add_column(index=index, name='DL', data=DL)
cs82_cat.add_column(index=index, name='DS', data=DS)

# Now you have what you need for Shear and Kappa
cs82_cat.write_to('gx_CS82.fits')
```

## Catalogue Format

### KiDS
Files are in HDF5 format. Tables are grouped as follows:

 - **coordinates**: ID, RAJ2000, DECJ2000, Z_B, Z_B_MIN, Z_B_MAX, T_B, ODDS, SG_FLAG

 - **lensing**:  ID, PSF_Q11, PSF_Q12, PSF_Q22, PSF_Strehl_ratio, PSF_e1, PSF_e1_exp1, PSF_e1_exp2, PSF_e1_exp3, PSF_e1_exp4, PSF_e1_exp5, PSF_e2, PSF_e2_exp1, PSF_e2_exp2, PSF_e2_exp3, PSF_e2_exp4, PSF_e2_exp5, bias_corrected_scalelength, bulge_fraction, contamination_radius, e1, e2, m, weight, fitclass, model_SNratio, model_flux, n_exposures_used, pixel_SNratio

 - **magnitudes**:  ID, CLASS_STAR, FLUX_RADIUS, FWHM_IMAGE, FWHM_WORLD, Flag, KRON_RADIUS, MAGERR_g, MAGERR_i, MAGERR_r, MAGERR_u, MAG_LIM_g, MAG_LIM_i, MAG_LIM_r, MAG_LIM_u, MAG_g, MAG_i, MAG_r, MAG_u, Xpos, Ypos, ZPT_offset

 - **others**: ID, KIDS_TILE, MASK, Patch, SeqNr, THELI_NAME

### CS82
Files are in HDF5 format. Tables are grouped as follows:

 - **coordinates**: 

 - **lensing**:  

 - **magnitudes**:  

 - **others**: 
 
### CFHTLens
Files are in HDF5 format. Tables are grouped as follows:

 - **coordinates**: 

 - **lensing**:  

 - **magnitudes**:  

 - **others**: 
