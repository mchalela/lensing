import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ; plt.ion()
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from scipy import fftpack, ndimage

from .. import gentools, Shear 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)



def read_map(file):
    ''' Read profile written with Map.write_to() method
    '''
    with fits.open(file) as f:
        mp = Map()
        mp.px = f['px'].data
        mp.py = f['py'].data
        mp.kappa = f['kappaE'].data + 1j*f['kappaB'].data
        mp.N = f['N'].data
        mp.shear_map = Shear.Map()
        mp.shear_map.shear = f['shear'].data
        mp.shear_map.beta = f['beta'].data
        mp.shear_map.stat_error = f['stat_error'].data
        mp.shear_map.px = f['px'].data
        mp.shear_map.py = f['py'].data
        # Complete variables
        mp.shear_map.shearx = mp.shear_map.shear * np.cos(mp.shear_map.beta)
        mp.shear_map.sheary = mp.shear_map.shear * np.sin(mp.shear_map.beta)
        mp.shear_map.shear1 = mp.shear_map.shear * np.cos(2*mp.shear_map.beta)
        mp.shear_map.shear2 = mp.shear_map.shear * np.sin(2*mp.shear_map.beta)
        dx = mp.px[0,1]-mp.px[0,0]
        mp.bin_size = (dx, dx)
    return mp


class Map(object):
    '''
    Kaiser-Squires reconstruction
    We follow the equations from Jeffrey et al 2018, section 2.2
    arxiv.org/pdf/1801.08945.pdf
    '''    
    def __init__(self, nbins=None, box_size=None, cosmo=None, back_dz=None):
    
        self.nbins = nbins
        self.cosmo = cosmo
        self.back_dz = back_dz
        self.box_size = box_size
        self._T_Dconj = None

        self.px, self.py = None, None
        self.kappa = None
        self.shear_map = None
        self.N = None
        self.bin_size = None

    def __getitem__(self, key):
        return getattr(self, key)


    def _kappa_map(self, shear_map):
        # Equations from Jeffrey 2018, section 2.2
        # Fourier transform of the shear field
        T_shear = fftpack.fft2( shear_map.shear1 + 1j*shear_map.shear2 )

        # Compute conjugate inversion kernel
        # Compute one time and save it, useful for error map
        if self._T_Dconj is None:
            self._T_Dconj = self._conjugate_inversion_kernel()

        # Compute kappa in fourier space and inverse transform
        T_kappa = T_shear * self._T_Dconj
        kappa_map = fftpack.ifft2(T_kappa)
        return kappa_map

    def _conjugate_inversion_kernel(self):
        ''' Define fourier grid for the kernel inversion
        '''
        dx = self.bin_size[0]
        dy = self.bin_size[1]
        nbins = self.px.shape[0]

        # create fourier grid
        k_x0, k_y0 = fftpack.fftfreq(nbins, d=dx), fftpack.fftfreq(nbins, d=dy)
        kx, ky = np.meshgrid(k_x0, k_y0, indexing='xy')

        T_Dconj = (kx**2 - ky**2 - 2j*kx*ky) / (kx**2 + ky**2)  # for k!=0
        T_Dconj[0, 0] = 0. + 0j        # for k=0 
        return T_Dconj

    # resize and smooth the image
    def _resize_and_smooth(self, km, resize, sigma_pix, truncate):
        km = ndimage.zoom(km, zoom=resize, order=0)
        km = ndimage.gaussian_filter(km, sigma=sigma_pix*resize, truncate=truncate)
        return km

    def gaussian_filter(self, sigma=10., truncate=5, resize=1, apply_to_err=False):
        ''' Apply gaussian filter to reduce high frequency noise
        resize is good for smooth plots, recomended: resize=100
        '''
        dx = self.bin_size[0]    # pixel size in Mpc/h or sacled
        sigma_pix = sigma/dx

        kE = self._resize_and_smooth(self.kappa.real, resize, sigma_pix, truncate)
        kB = self._resize_and_smooth(self.kappa.imag, resize, sigma_pix, truncate)
        smooth_kappa = kE + 1j*kB

        if apply_to_err:
            kE_err = self._resize_and_smooth(self.kappa_err.real, resize, sigma_pix, truncate)
            kB_err = self._resize_and_smooth(self.kappa_err.imag, resize, sigma_pix, truncate)
            smooth_kappa_err = kE_err + 1j*kB_err
            return smooth_kappa, smooth_kappa_err
        else:
            return smooth_kappa

    def _plot(self, mE, mB, title, cmap):
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap(cmap)
        cmap.set_bad(color='k')

        extent = [self.px.min(), self.px.max(),self.py.min(), self.py.max()]
        vmin, vmax = mE.min(), mE.max()

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set(aspect='equal')
        im = ax[0].imshow(mE, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax[0].set_xlabel('$x\,[Mpc/h]$', fontsize=12)
        ax[0].set_ylabel('$y\,[Mpc/h]$', fontsize=12)
        ax[0].set_title('E-mode', fontsize=12)

        ax[1].set(aspect='equal')
        ax[1].imshow(mB, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax[1].set_xlabel('$x\,[Mpc/h]$', fontsize=12)
        ax[1].set_title('B-mode', fontsize=12)

        cbar = fig.colorbar(im,  ax=ax.ravel().tolist(), orientation='horizontal')
        cbar.ax.set_xlabel(r'$\mathrm{\Sigma\,[\,h\,M_{\odot}\,pc^{-2}\,]}$', fontsize=12)
        plt.suptitle(title, fontsize=14)
        plt.show()
        return None

    def QuickPlot(self, sigma=0., resize=1, plot_err=False, cmap='jet'):
        ''' Plot the reconstructed kappa map.
        '''
        if sigma>0:
            if plot_err:
                k, k_err = self.gaussian_filter(sigma, resize=resize, apply_to_err=True)
                kE, kB = k.real, k.imag
                kE_err, kB_err = k_err.real, k_err.imag
            else:
                k = self.gaussian_filter(sigma, resize=resize)
                kE, kB = k.real, k.imag                
        else:
            if plot_err:
                kE, kB = self.kappa.real, self.kappa.imag
                kE_err, kB_err = self.kappa_err.real, self.kappa_err.imag
            else:
                kE, kB = self.kappa.real, self.kappa.imag


        self._plot(kE, kB, 'Convergence Map', cmap=cmap)
        if plot_err:
            self._plot(kE_err, kB_err, 'Convergence Map Error', cmap=cmap)
        return None

    def write_to(self, file, header=None, overwrite=False):
        '''Write Map to a FITS file.
        Add a header to lensing.shear.Profile output file
        to know the sample parameters used to build it.
        
         file:      (str) Name of output file
         header:    (dic) Dictionary with parameter cuts. Optional.
                Example: {'z_min':0.1, 'z_max':0.3, 'odds_min':0.5}
         overwrite: (bool) Flag to overwrite file if it already exists.
        '''
        if os.path.isfile(file):
            if overwrite:
                os.remove(file)
            else:
                raise IOError('File already exist. You may want to use overwrite=True.')
        
        hdulist = [fits.PrimaryHDU()]
        hdulist.append(fits.ImageHDU(self.px, name='px'))
        hdulist.append(fits.ImageHDU(self.py, name='py'))
        hdulist.append(fits.ImageHDU(self.kappa.real, name='kappaE'))
        hdulist.append(fits.ImageHDU(self.kappa.imag, name='kappaB'))
        hdulist.append(fits.ImageHDU(self.N, name='N'))
        # Save shear map data
        hdulist.append(fits.ImageHDU(self.shear_map.shear, name='shear'))
        hdulist.append(fits.ImageHDU(self.shear_map.beta, name='beta'))
        hdulist.append(fits.ImageHDU(self.shear_map.stat_error, name='stat_error'))
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(file, overwrite=overwrite)
        return None



@gentools.timer
class KappaMap(Map):
    '''
    Kaiser-Squires reconstruction
    We follow the equations from Jeffrey et al 2018, section 2.2
    arxiv.org/pdf/1801.08945.pdf
    '''

    def __init__(self, data_L, data_S, scale=None, nbins=None, box_size=None, nboot=0, cosmo=cosmo,
        back_dz=0.1, precomputed_distances=True, njobs=1, mirror=None, rotate=None, colnames=None):

        super().__init__(nbins=nbins, box_size=box_size, cosmo=cosmo, back_dz=back_dz)

        self.scale = scale
        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.nboot = nboot
        self.mirror = mirror
        self.rotate = rotate

        if colnames is None: colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}
        self.colnames = colnames

        # Compute shear map
        shear_map = self._shear_map(data_L, data_S, save_shear_map=True)
        # Compute kappa and error map from the shear map
        self.kappa = self._kappa_map(shear_map)
        if self.nboot>0:
            self.kappa_err = self._error_map(data_L, data_S)

    def _shear_map(self, data_L, data_S, save_shear_map=True):

        shear_map = Shear.ShearMap(data_L=data_L, data_S=data_S, scale=self.scale, nbins=self.nbins, 
            mirror=self.mirror, box_size=self.box_size, colnames=self.colnames,
            cosmo=self.cosmo, back_dz=self.back_dz, rotate=self.rotate,
            precomputed_distances=self.precomputed_distances, njobs=self.njobs)

        # Save shear map for reference
        if save_shear_map:
            px = shear_map.px      # in Mpc/h
            py = shear_map.py      # in Mpc/h
            dx = px[0,1]-px[0,0]  
            dy = py[1,0]-py[0,0]
            bin_size = (dx, dy)    # in Mpc/h
            self.shear_map = shear_map
            self.px, self.py = px, py      # in Mpc/h
            self.N = self.shear_map.N
            self.bin_size = bin_size
        return shear_map      

    def _error_map(self, data_L, data_S):

        if self.box_size_scaled is None:
                raise ValueError('You need to specify a box_size_scaled value '
                    'for the bootstrap analysis.')

        cube_shape = self.kappa.shape + (self.nboot, )    # add extra dimension for each map
        kE_err_cube = np.zeros(cube_shape)
        kB_err_cube = np.zeros(cube_shape)

        index = np.arange(len(data_L))
        with NumpyRNGContext(seed=1):
            index_boot = bootstrap(index, self.nboot).astype(int)

        # This for does not need parallelization because _shear_map is already parallelized
        for i in range(self.nboot):
            if isinstance(data_L, pd.DataFrame):
                b_data_L = data_L.iloc[index_boot[i]]
            else:
                b_data_L = data_L[index_boot[i]]         # assuming numpy array

            b_shear = self._shear_map(b_data_L, data_S, save_shear_map=False)
            b_kappa = self._kappa_map(b_shear)
            kE_err_cube[:, :, i] = b_kappa.real
            kB_err_cube[:, :, i] = b_kappa.imag

        kE_err = np.std(kE_err_cube, axis=2)
        kB_err = np.std(kB_err_cube, axis=2)
        error_map = kE_err + 1j*kB_err
        return error_map




class fromShearMap(Map):
    '''
    Kaiser-Squires reconstruction
    We follow the equations from Jeffrey et al 2018, section 2.2
    arxiv.org/pdf/1801.08945.pdf
    '''
    def __init__(self, shear_map):

        super().__init__()

        self.shear_map = shear_map
        self.N = self.shear_map.N
        self.px = shear_map.px      # in Mpc/h
        self.py = shear_map.py      # in Mpc/h
        dx = self.px[0, 1] - self.px[0, 0]  
        dy = self.py[1, 0] - self.py[0, 0]
        bin_size = (dx, dy)    # in Mpc/h
        self.bin_size = bin_size


        # Compute kappa and error map from the shear map
        self.kappa = self._kappa_map(self.shear_map)
        #if self.nboot>0:
        #    self.kappa_err = self._error_map(data_L, data_S)


#################################################################################################
# Deprecated

@gentools.timer
class ExpandedMap(Map):
    '''
    Kaiser-Squires reconstruction
    We follow the equations from Jeffrey et al 2018, section 2.2
    arxiv.org/pdf/1801.08945.pdf
    '''

    def __init__(self, data=None, nbins=None, box_size_hMpc=None, nboot=0, cosmo=cosmo,
        back_dz=0.1, precomputed_distances=True, njobs=1):

        super().__init__(nbins=nbins, box_size_hMpc=box_size_hMpc, cosmo=cosmo, back_dz=back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.nboot = nboot

        # Compute shear map
        shear_map = self._shear_map(data, save_shear_map=True)

        # Compute kappa and error map
        self.kappa = self._kappa_map(shear_map)
        if self.nboot>0:
            self.kappa_err = self._error_map(data)

    def _shear_map(self, data, save_shear_map=True):

        shear_map = Shear.ExpandedMap(data=data, nbins=self.nbins, 
                                box_size_hMpc=self.box_size_hMpc, cosmo=self.cosmo, back_dz=self.back_dz, 
                                precomputed_distances=self.precomputed_distances, njobs=self.njobs)
        # Save shear map for reference
        if save_shear_map:
            px = shear_map.px      # in Mpc/h
            py = shear_map.py      # in Mpc/h
            dx = px[0,1]-px[0,0]  
            dy = py[1,0]-py[0,0]
            bin_size = (dx, dy)    # in Mpc/h
            self.shear_map = shear_map
            self.px, self.py = px, py      # in Mpc/h
            self.N = self.shear_map.N
            self.bin_size = bin_size
        return shear_map  

    def _error_map(self, data):

        if self.box_size_hMpc is None:
                raise ValueError('You need to specify a box_size_hMpc value '
                    'for the bootstrap analysis.')

        cube_shape = self.kappa.shape + (self.nboot, )    # add extra dimension for each map
        kE_err_cube = np.zeros(cube_shape)
        kB_err_cube = np.zeros(cube_shape)

        index = np.arange(len(data))
        with NumpyRNGContext(seed=1):
            index_boot = bootstrap(index, self.nboot).astype(int)

        for i in range(self.nboot):
            if isinstance(data, pd.DataFrame):
                b_data = data.iloc[i]
            else:
                b_data = data[i]         # assuming numpy array

            b_shear = self._shear_map(b_data, save_shear_map=False)
            b_kappa = self._kappa_map(b_shear)
            kE_err_cube[:, :, i] = b_kappa.real
            kB_err_cube[:, :, i] = b_kappa.imag

        kE_err = np.std(kE_err_cube, axis=2)
        kB_err = np.std(kB_err_cube, axis=2)
        error_map = kE_err + 1j*kB_err
        return error_map