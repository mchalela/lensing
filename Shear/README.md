# Shear: Module to compute radial profiles and 2D maps

## Profile

To compute the shear profile this module uses the data DataFrame from the LensCat catalogue. Let's say you have already generated the catalogue and saved it to a file.

```python
from lensing import LensCat, Shear
cat = LensCat.Catalog.read_catalog('gx_CFHT.fits')
```

Now you can play with your data to make different quality cuts, for exaple:
```python
delta_z = 0.1
odds_min = 0.5
zback_max = 1.3

mask = (cat.data['Z_B']>(cat.data['z']+delta_z)) & (cat.data['ODDS']>=odds_min)
mask *= (cat.data['Z_B']<zback_max)
```

Now to compute the profile
```python
profile = Shear.Profile(cat.data[mask], rin_hMpc=0.1, rout_hMpc=10., bins=30)
```
This will return a profile object with 30 logarithmic bins (space='lin' for linear bins). You can use the keyword boot_n=100 to compute 100 bootstrap resampling to compute the errors, otherwise you get poisson statistical errors.

**You can now fit your model!**

 You can save your profile to a file with:
 ```python
 profile.write_to('CFHT.profile') 
 ```
 And then read it latter as a profile object with:
 ```python
 profile = Shear.Profile.read_profile('CFHT.profile')
 ```

 This is an example of the redMaPPer shear profiles for the three lensing catalogs. The clusters are stacked in the redshift range z = (0.1, 0.33).
 ![Shear profiles](https://github.com/mchalela/lensing/blob/master/Shear/shear_profiles.png)

## ShearMap
