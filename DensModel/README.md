# DensModel: Module for density model fitting
This module defines different mass models and fitting methods to fit the Shear profiles

## Density
Theroetical density contrast radial functions. First you need to initialize a Density() object with a given redshift and cosmology.

```python
import numpy as np
from lensing import DensModel
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# Initialize the Density object. Default cosmology is Planck15.
DN = DensModel.Density(z=0.3, cosmo=cosmo)

# Now you can create a theoretical model at this radial distances
r_h = np.geomspace(0.1, 1, 50)    # distance in Mpc/h
```
### Point Mass
A point mass profile is generally used to model the central barionc component, the BCG. It can be parametrized with the stellar mass Mstar or withe the M200 halo mass using the scaling relation of Johnston et a. 2007 (arxiv.org/pdf/0709.1159.pdf). In both cases the mass is in log10 and in Msun/h.
```python
bcg1 = DN.BCG(r_h, logMstar_h=13.)
bcg2 = DN.BCG_with_M200(r_h, logM200_h=13.)
```

### Singular Isothermal Sphere
This profile is parametrized with the isotropic velocity dispersion in km/s.
```python
sis = DN.SIS(r_h, disp=300.)
```

### Navarro, Frenk & White
The NFW profile for the shape of the dark matter halo. The usual profile, NFW() is parametrized with M200c and the concentration parameter is fixed with the scaling relation of Duffy et al 2008 (arxiv.org/pdf/0804.2486.pdf).
You can also compute the profile with an offset between the profile center and the dark matter center. NFWoff() computes the NFW density contrast profile with a gaussian miscentered kernel, where the gaussian sigma, disp_offset_h, must be in Mpc/h. In both cases the M200_h must be in Msun/h.
The NFWCombined() is sum of the centered and miscentered components with the p_cen weight, i.e.: p_cen*NFW() + (1-p_cen)*NFWoff()
```python
nfw = DN.NFW(r_h, logM200_h=13.)
nfw_off = DN.NFWoff(r_h, logM200_h=13., disp_offset_h=0.1)
nfw_comb = DN.NFWCombined(r_h, logM200_h=13., disp_offset_h=0.1, p_cen=1.)
```

### Second Halo Term
For this you need to install the **camb** package to compute the power spectrum.
This computes the second halo contribution to the shear signal. It is parametrized with the halo M200_h and the delta parameter above the medium density of the universe. Usually Delta=200 is a good default value.
```python
second_halo = DN.SecondHalo(r_h, logM200_h=13., Delta=200.)
```

## DensityModel
This class is simply a wrapper of the Density() class to return an astropy model object useful for fitting your data with one or more of the theoretical models using the Fitter() class.
```python
# Initialize a DensityModel() instance for a given redshift and cosmology.
# This will initialize a Density() instance with these parameters.
DNM = DensModel.DensityModel(z=0.3, cosmo=cosmo)
```

Now you can create fittable models and add them to create a compund model. So, lets say you want to fit your data with the BCG, NFW centered and miscentered, and the Second Halo term.
```python
# Initialize your individual components
bcg_model = DNM.BCG_with_M200()
nfwcomb_model = DNM.NFWCombined()   # Use this to fit both NFW components
shalo_model = DNM.SecondHalo()

# Now add them to create your compund model
shear_model = DNM.AddModels([bcg_model, nfwcomb_model, shalo_model])
```
And that's it! You have your model with the M200_h mass tied for your different models so it will fit a single mass.

## Fitter
To fit your model to your data you can use this class. For the moment it has two fitting procedures implemented depending on how timeconsuming is the computation of your model.

```python
# First initialize a Fitter object with you data, model and initial parameters
start_params = [13., 0.2, 0.7]  # This is in order: logM200_h=13, disp_offset=0.2, p_cen=0.7

# Lets say you computed your profile object with Shear.Profile()
fitter = DensModel.Fitter(r=profile.r_hMpc,
                          shear=profile.shear, 
                          shear_err=profile.shear_error, 
			  model=shear_model, 
                          start_params=start_params)
```


