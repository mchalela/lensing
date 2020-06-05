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
df = pd.read_csv('path/to/lens/catalogue.csv')

# We need the search radius in degrees, so we use Mpc2deg() from the gentools module.
R_Mpc = 1.
R_deg = gentools.Mpc2deg(R_Mpc=R_Mpc, z=df['z'], cosmo=cosmo)

# Search for galaxies in the survey you want, and append your lens catalogue.
# You can lunch a parallel search in njobs.
cs82_cat = LensCat.CS82.find_neighbors(centre=df[['RA','DEC']], upper_radii=R_deg, append_data=df, compressed=True)
kids_cat = LensCat.KiDS.find_neighbors(centre=df[['RA','DEC']], upper_radii=R_deg, append_data=df, compressed=True)
cfht_cat = LensCat.CFHT.find_neighbors(centre=df[['RA','DEC']], upper_radii=R_deg, append_data=df, compressed=True)
```

And thats it! Take note of the compressed=True argument. When set to True the resulting catalogue will have two catalogues within it, named .data_L and .data_S. If set to False it will only make one expanded catalogue named .data. For larga catalogues it's better to set it as True and save a lot of memory usage.

Now you can save your individual catalogues.
```python
cs82_cat.write_to('gx_CS82.fits')
kids_cat.write_to('gx_KiDS.fits')
cfht_cat.write_to('gx_CFHT.fits')
```

Or, you can combine them and then save the file. There are two operators defined for compressed catalogues: and '&', and sum '+'. Combining with '&' will result in a direct concatenation of catalogues. Combining with '+' will discard repeated lenses priorizing the catalog on the left side. For example:
```python
cat = cs82_cat + kids_cat + cfht_cat
cat.write_to('gx_'+cat.name+'.fits')
```
This will combine catalogues checking removing lenses from kids (if it is in cs82) and from cfht (if it is in kids or cs82).

### Load different columns
That was the easy way. By default, if you search for galaxies directly with the **find_neighbors()** method you will load only these columns from the catalogues: RAJ2000, DECJ2000, Z_B, e1, e2, m, weight, ODDS, fitclass, MASK. And in particular, the CFHTLens catalogue will also load the c2 additive bias column, correct the e2 component of ellipticity and drop the c2 column from the catalogue, thats why you wont see it.

This is because the catalogue needs to be loaded in memory first with the **load(fields=None, columns=None, science_cut=True)** method. If you don't load the catalogue before searching for galaxies, it will be loaded automatically with those default columns. You can control what columns and fields are loaded.
```python
# Lets say you want from the CS82 those columns and also the BPZ_LOW95
columns = ['RAJ2000','DECJ2000','Z_B','e1','e2','m','weight','ODDS','fitclass','MASK', 'BPZ_LOW95']
LensCat.CS82.load(columns=columns)
cs82_cat = LensCat.CS82.find_neighbors(centre=df[['RA','DEC']], upper_radii=R_deg, append_data=df, njobs=40)

# Or lets say you want from the KiDS catalogue the default columns, but
# only in the G9 and G12 fields.
LensCat.KiDS.load(fields=['G9','G12'])
kids_cat = LensCat.KiDS.find_neighbors(centre=df[['RA','DEC']], upper_radii=R_deg, append_data=df, njobs=40)
```

The science_cut will usually be True, and it gives you all the galaxies with **fitclass=0**, **MASK<=1** and **weight>0**. So if you want these cut you need to load these columns. For the CFHT in particular, the c2 column must be loaded.

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
