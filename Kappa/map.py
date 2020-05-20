import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ; plt.ion()
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from scipy import fftpack, ndimage
from .. import gentools, Shear 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)


class KappaMap(object):
    '''
    Kaiser-Squires reconstruction
    We follow the equations from Jeffrey et al 2018, section 2.2
    arxiv.org/pdf/1801.08945.pdf
    '''

    def __init__(self, data=None, nbins=None, box_size_hMpc=None, nboot=0, cosmo=cosmo):

        if data is None:
            raise ValueError('KappaMap needs some data to work with...')

        # Compute kappa and error map
        self.kappa = self._kappa_map(data, nbins, box_size_hMpc, cosmo)
        if nboot>0:
            self.kappa_err = self._error_map(nboot, data, nbins, box_size_hMpc, cosmo)


    def _kappa_map(self, data, nbins, box_size_hMpc, cosmo, save_shear_map=True):

        # Compute the shear map
        shear_map = Shear.ShearMap(data=data, nbins=nbins, 
                                    box_size_hMpc=box_size_hMpc, cosmo=cosmo)

        px = shear_map.px      # in Mpc/h
        py = shear_map.py      # in Mpc/h
        dx = px[0,1]-px[0,0]  
        dy = py[1,0]-py[0,0]
        bin_size = (dx, dy)    # in Mpc/h

        # Save shear map for reference
        if save_shear_map:
            self.shear_map = shear_map
            self.px, self.py = px, py      # in Mpc/h
            self.bin_size = bin_size

        # Equations from Jeffrey 2018, section 2.2
        # Fourier transform of the shear field
        T_shear = fftpack.fft2( shear_map.e1 + 1j*shear_map.e2 )

        # Compute conjugate inversion kernel
        T_Dconj = self._conjugate_inversion_kernel(nbins)

        # Compute kappa in fourier space and inverse transform
        T_kappa = T_shear * T_Dconj
        kappa_map = fftpack.ifft2(T_kappa)

        return kappa_map


    def _conjugate_inversion_kernel(self, nbins):
        ''' Define fourier grid for the kernel inversion
        '''
        dx = self.bin_size[0]
        dy = self.bin_size[1]

        # create fourier grid
        k_x0, k_y0 = fftpack.fftfreq(nbins, d=dx), fftpack.fftfreq(nbins, d=dy)
        kx, ky = np.meshgrid(k_x0, k_y0, indexing='xy')

        T_Dconj = (kx**2 - ky**2 - 2j*kx*ky) / (kx**2 + ky**2)  # for k!=0
        T_Dconj[0, 0] = 0. + 0j        # for k=0 
        return T_Dconj

    def _error_map(self, boot_n, data, nbins, box_size_hMpc, cosmo):

        if box_size_hMpc is None:
                raise ValueError('You need to specify a box_size_hMpc value '
                    'for the bootstrap analysis.')

        cube_shape = self.kappa.shape + (boot_n, )    # add extra dimension for each map
        kE_err_cube = np.zeros(cube_shape)
        kB_err_cube = np.zeros(cube_shape)

        index = np.arange(len(data))
        with NumpyRNGContext(seed=1):
            index_boot = bootstrap(index, boot_n).astype(int)

        for i in range(boot_n):
            if isinstance(data, pd.DataFrame):
                b_data = data.iloc[i]
            else:
                b_data = data[i]         # assuming numpy array

            b_kappa = self._kappa_map(b_data, nbins, box_size_hMpc, cosmo, save_shear_map=False)
            kE_err_cube[:, :, i] = b_kappa.real
            kB_err_cube[:, :, i] = b_kappa.imag

        kE_err = np.std(kE_err_cube, axis=2)
        kB_err = np.std(kB_err_cube, axis=2)
        error_map = kE_err + 1j*kB_err
        return error_map

    # resize and smooth the image
    def _resize_and_smooth(km):
        km = ndimage.zoom(km, zoom=resize, order=0)
        km = ndimage.gaussian_filter(km, sigma=sigma_pix*resize, truncate=truncate)
        return km

    def gaussian_filter(self, sigma_hkpc=10., truncate=5, resize=1, apply_to_err=False):
        ''' Apply gaussian filter to reduce high frequency noise
        resize is good for smooth plots, recomended: resize=100
        '''
        dx = self.bin_size[0]    # pixel size in Mpc/h
        sigma_hMpc = sigma_hkpc * 1e-3
        sigma_pix = sigma_hMpc/dx


        kE = _resize_and_smooth(self.kappa.real)
        kB = _resize_and_smooth(self.kappa.imag)
        smooth_kappa = kE + 1j*kB

        if apply_to_err:
            kE_err = _resize_and_smooth(self.kappa_err.real)
            kB_err = _resize_and_smooth(self.kappa_err.imag)
            smooth_kappa_err = kE_err + 1j*kB_err
            return smooth_kappa, smooth_kappa_err
        else:
            return smooth_kappa

    def _plot(mE, mB, title):
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

    def QuickPlot(self, sigma_hkpc=0., resize=1, plot_err=False, cmap='jet'):
        ''' Plot the reconstructed kappa map.
        '''

        if sigma_hkpc>0:
            if plot_err:
                k, k_err = self.gaussian_filter(sigma_hkpc, resize=resize, apply_to_err=True)
                kE, kB = k.real, k.imag
                kE_err, kB_err = k_err.real, k_err.imag
            else:
                k = self.gaussian_filter(sigma_hkpc, resize=resize)
                kE, kB = k.real, k.imag                
        else:
            if plot_err:
                kE, kB = self.kappa.real, self.kappa.imag
                kE_err, kB_err = self.kappa_err.real, self.kappa_err.imag
            else:
                kE, kB = self.kappa.real, self.kappa.imag


        _plot(kE, kB, 'Convergence Map')
        if plot_err:
            _plot(kE_err, kB_err, 'Convergence Map Error')
        return None
