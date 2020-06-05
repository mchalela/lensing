# Kappa: Module to compute 2D density maps

## Map
To compute the shear map this module uses the data_L, data_S DataFrame from the LensCat catalogue. Let's say you have already generated the catalogue and saved it to a file.

```python
from lensing import LensCat, Kappa
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
kmp = Kappa.CompressedMap(data_L=data_L[mask_L], data_S=data_S[mask_S], nbins=10, box_size_hMpc=0.5, 
precomputed_distances=True, mirror='x', njobs=5, dz_back=0.1, nboot=100)
```
This will compute in parallel with 5 threads, and return a map object with a 10x10 grid of side length equal to box_size_hMpc. The keyword precomputed_distances=True will interpolate the lensing distances DL,DS,DLS from a previously saved file. The keyword mirror combines the data reflecting in the indicated axis. For example: mirror='y' will combine the sources of -y as if they where in +y and the resulting map will be reflected. This is simply to take advantage of your system simetry. The keyword dz_back will consider sources that have a redshift larger than its lens, i.e.: z_L > z_S + dz_back.

 You can save your map to a file with:
 ```python
 kmp.write_to('kmp_CFHT.fits') 
 ```
 And then read it latter as a profile object with:
 ```python
 kmp = Kappa.read_map('smp_CFHT.fits')
 ```

 This is an example of the redMaPPer convergence map for the four lensing catalogs. The clusters are stacked in the redshift range z = (0.1, 0.33).
 ![Convergence maps](https://github.com/mchalela/lensing/blob/master/Shear/shear_maps.png)
