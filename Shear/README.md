# Shear: Module to compute radial profiles and 2D maps

## Profile

To compute the shear profile this module uses the data_L, data_S DataFrame from the LensCat catalogue. Let's say you have already generated the catalogue and saved it to a file.

```python
from lensing import LensCat, Shear
cat = LensCat.read_catalog('gx_CFHT.fits')
data_L = cat.data_L
data_S = cat.data_S
```

Now you can play with your data to make different quality cuts, for exaple:
```python
odds_min = 0.5
zback_max = 1.3

mask_S = (data_S['ODDS']>=odds_min) & (data_S['Z_B']<zback_max)
mask_L = data_L['prop1']>some_value
```

Now to compute the profile
```python
spf = Shear.CompressedProfile(data_L=data_L[mask_L], data_S=data_S[mask_S], rin_hMpc=0.1, rout_hMpc=10., bins=30, njobs=5, dz_back=0.1)
```
This will compute in parallel with 5 threads, and return a profile object with 30 logarithmic bins (space='lin' for linear bins). You can use the keyword nboot=100 to compute 100 bootstrap resampling to compute the errors, otherwise you get poisson statistical errors. The keyword dz_back will consider sources that have a redshift larger than its lens, i.e.: z_L > z_S + dz_back.

**You can now fit your model!**

 You can save your profile to a file with:
 ```python
 spf.write_to('CFHT.profile') 
 ```
 And then read it latter as a profile object with:
 ```python
 spf = Shear.read_profile('CFHT.profile')
 ```

 This is an example of the redMaPPer shear profiles for the three lensing catalogs. The clusters are stacked in the redshift range z = (0.1, 0.33).
 ![Shear profiles](https://github.com/mchalela/lensing/blob/master/Shear/shear_profiles.png)

## ShearMap
To compute the shear map this module uses the data_L, data_S DataFrame from the LensCat catalogue. Let's say you have already generated the catalogue and saved it to a file.

```python
from lensing import LensCat, Shear
cat = LensCat.read_catalog('gx_CFHT.fits')
data_L = cat.data_L
data_S = cat.data_S
```

Now you can play with your data to make different quality cuts, for exaple:
```python
odds_min = 0.5
zback_max = 1.3

mask_S = (data_S['ODDS']>=odds_min) & (data_S['Z_B']<zback_max)
mask_L = data_L['prop1']>some_value
```

Now to compute the map
```python
smap = Shear.CompressedMap(data_L=data_L[mask_L], data_S=data_S[mask_S], nbins=10, box_size_hMpc=0.5, 
precomputed_distances=True, mirror='x', njobs=5, dz_back=0.1)
```
This will compute in parallel with 5 threads, and return a map object with a 10x10 grid of side length equal to box_size_hMpc. The keyword precomputed_distances=True will interpolate the lensing distances DL,DS,DLS from a previously saved file. The keyword mirror combines the data reflecting in the indicated axis. For example: mirror='y' will combine the sources of -y as if they where in +y and the resulting map will be reflected. This is simply to take advantage of your system simetry. The keyword dz_back will consider sources that have a redshift larger than its lens, i.e.: z_L > z_S + dz_back.

 You can save your map to a file with:
 ```python
 smp.write_to('smp_CFHT.fits') 
 ```
 And then read it latter as a profile object with:
 ```python
 smp = Shear.read_map('smp_CFHT.fits')
 ```

 This is an example of the redMaPPer shear map for the four lensing catalogs. The clusters are stacked in the redshift range z = (0.1, 0.33).
 ![Shear maps](https://github.com/mchalela/lensing/blob/master/Shear/shear_maps.png)
