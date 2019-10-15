# DensModel: Module for density model fitting
This module defines different mass models and fitting methods to fit the Shear profiles

## Density
Theroetical radial density functions. First you need to initialize a Density() object with a given redshift and cosmology.

```python
import numpy as np
from lensing import DensModel
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# Initialize the Density object. Default cosmology is Planck15.
DN = DensModel.Density(z=0.3, cosmo=cosmo)

# Now you can create a theoretical model
r_h = np.geomspace(0.1, 1, 20)    # distance in Mpc/h
```
### Point Mass
A point mass profile is generally used to model the central barionc component, the BCG.
```python
bcg1 = DN.BCG(r_h, logMstar_h=13.)
bcg2 = DN.BCG_with_M200(r_h, logM200_h=13.)
```

### Singular Isothermal Sphere
```python
sis = DN.SIS(r_h, disp=300.)
```

### Navarro, Frenk & White
```python
nfw = DN.NFW(r_h, logM200_h=13.)
nfw_off = DN.NFWoff(r_h, logM200_h=13., disp_offset_h=0.1)
nfw_comb = DN.NFWCombined(r_h, logM200_h=13., disp_offset_h=0.1, p_cen=1.)
```

### Second Halo Term
```python
second_halo = DN.SecondHalo(r_h, logM200_h=13., Delta=200.)
```

## DensityModel

## Fitter
