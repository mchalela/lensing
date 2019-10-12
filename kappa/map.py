import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ; plt.ion()
#from astropy.stats import bootstrap
#from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from lensing import gentools, shear 

from scipy import fftpack, ndimage
from mpl_toolkits.mplot3d import Axes3D

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

    def __init__(self, data=None, nbins=None, gals_per_bins=50., box_size_hMpc=None, cosmo=cosmo):

        if data is None:
            raise ValueError('KappaMap needs some data to work with...')

        # Set nbins
        if nbins is None:
            nbins = int(nGalaxies / np.sqrt(gals_per_bins))

        # Compute the shear map
        shear_map = shear.ShearMap(data=data, nbins=nbins, 
            gals_per_bins=gals_per_bins, box_size_hMpc=box_size_hMpc, cosmo=cosmo)
        # save the shear map for reference
        self.shear_map = shear_map

        self.px = shear_map.px      # in Mpc/h
        self.py = shear_map.px      # in Mpc/h
        e1_map = shear_map.e1     # not sure about this transpose
        e2_map = shear_map.e2     # not sure about this transpose

        # Equations from Jeffrey 2018, section 2.2
        # Fourier transform of the shear field
        T_shear = fftpack.fft2( e1_map + 1j*e2_map )

        # Compute conjugate inversion kernel
        T_Dconj = self._conjugate_inversion_kernel(nbins)

        # Compute kappa in fourier space and inverse transform
        T_kappa = T_shear * T_Dconj
        self.kappa = fftpack.ifft2(T_kappa) 


    def _conjugate_inversion_kernel(self, nbins):
        ''' Define fourier grid for the kernel inversion
        '''
        dx = self.px[1,0] - self.px[0,0]
        dy = self.py[1,0] - self.py[0,0]

        # create fourier grid
        k_x0, k_y0 = fftpack.fftfreq(nbins, d=dx), fftpack.fftfreq(nbins, d=dy)
        kx, ky = np.meshgrid(k_x0, k_y0, indexing='ij')

        T_Dconj = (kx**2 - ky**2 - 2j*kx*ky) / (kx**2 + ky**2)  # for k!=0
        T_Dconj[0, 0] = 0. + 0j        # for k=0 
        return T_Dconj

    def gaussian_filter(self, sigma_hkpc=10., truncate=5):
        ''' Apply gaussian filter to reduce high frequency noise
        '''
        dx = self.px[1,0] - self.px[0,0]    # pixel size in Mpc/h
        sigma_hMpc = sigma_hkpc * 1e-3
        sigma_pix = sigma_hMpc/dx

        # resize the image
        resize = 100
        kE = ndimage.zoom(np.real(self.kappa), zoom=resize, order=0)
        kB = ndimage.zoom(np.imag(self.kappa), zoom=resize, order=0)

        kE = ndimage.gaussian_filter(kE, sigma=sigma_pix*resize, truncate=truncate) 
        kB = ndimage.gaussian_filter(kB, sigma=sigma_pix*resize, truncate=truncate)

        smooth_kappa = kE + 1j*kB
        return smooth_kappa


    def QuickPlot(self, sigma_hkpc=0., kappa_mode='E', cmap=None):
        ''' Plot the reconstructed kappa map
        kappa_mode: 'E', 'B', 'EB'
        '''
        if cmap is None:
            from matplotlib import cm
            cmap=cm.jet


        if sigma_hkpc>0:
            k = self.gaussian_filter(sigma_hkpc)
            kE = np.real(k)
            kB = np.imag(k)
        else:
            kE = np.real(self.kappa)
            kB = np.imag(self.kappa)

        extent = [self.px.min(), self.px.max(),self.py.min(), self.py.max()]
        vmin, vmax = kE.min(), kE.max()

        if kappa_mode == 'E':
            plt.figure()
            plt.imshow(kE, extent=extent, cmap=cmap, origin='lower')
            plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
            plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
            plt.title('Convergence map E-mode', fontsize=14)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r'$\mathrm{\Delta\Sigma\,[\,h\,M_{\odot}\,pc^{-2}\,]}$', fontsize=14)
            plt.show()
        elif kappa_mode == 'B':
            plt.figure()
            plt.imshow(kB, extent=extent, cmap=cmap, origin='lower')
            plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
            plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
            plt.title('Convergence map B-mode', fontsize=14)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r'$\mathrm{\Delta\Sigma\,[\,h\,M_{\odot}\,pc^{-2}\,]}$', fontsize=14)
            plt.show()
        elif kappa_mode == 'EB':
            plt.figure()
            plt.subplot(121, aspect='equal') 
            plt.imshow(kE, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
            plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
            plt.title('Convergence map E-mode', fontsize=14)

            plt.subplot(122, aspect='equal')
            plt.imshow(kB, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
            plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
            plt.title('Convergence map B-mode', fontsize=14)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r'$\mathrm{\Delta\Sigma\,[\,h\,M_{\odot}\,pc^{-2}\,]}$', fontsize=14)
            plt.show()


'''
# Shear orientado tangencialmente sin ruido
def e_comp(x,y):

    r = (x**2+y**2)**0.5
    e = 1./(2*r)
    th= np.arctan2(y,x)  #revisar signo del angulo

    #1er cuadrante
    m_th = (y>0.)*(x>0.)
    th[m_th] = th[m_th] + np.pi/2.
    #2do cuadrante
    m_th = (y>0.)*(x<0.)
    th[m_th] = th[m_th] - np.pi/2.      
    #3er cuadrante
    m_th = (y<0.)*(x<0.)
    th[m_th] = np.pi*3./2. - abs(th[m_th])
    #4to cuadrante
    m_th = (y<0.)*(x>0.)
    th[m_th] = np.pi/2. - abs(th[m_th])

    # add noise to the angle
    #th += np.random.normal(0, np.pi/8, size=th.shape)

    e1 = e*np.cos(2*th)
    e2 = e*np.sin(2*th)

    #~ ex = e1 + e2*np.cos(np.pi/4.)
    #~ ey = e2*np.sin(np.pi/4.)
    ex = e*np.cos(th)
    ey = e*np.sin(th)   

    #return ex, ey, e
    return e1, e2, e



"""PLOTTING"""
fig = plt.figure()
plt.imshow(np.real(k))
fig = plt.figure()
plt.imshow(kE)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, np.real(k), cmap='viridis')
'''