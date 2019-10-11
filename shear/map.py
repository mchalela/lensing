import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from astropy.stats import bootstrap
#from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from lensing import gentools 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)


class ShearMap(object):

    def __init__(self, data=None, nbins=None, gals_per_bins=50., box_size_hMpc=None, cosmo=cosmo):

        if data is None:
            raise ValueError('ShearMap needs some data to work with...')

        # Define some parameters...
        nGalaxies = len(data)
        Mpc_scale = self.set_Mpc_scale(dl=data['DL'])

        # Compute distance and ellipticity components...
        dist, theta = gentools.sphere_angular_vector(data['RAJ2000'], data['DECJ2000'],
                                                    data['RA'], data['DEC'], units='deg')
        theta += 90. 
        dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # distance to the lens in Mpc/h

        # Transfrom e1,e2 to cartesian components ex,ey
        px = dist_hMpc * np.cos(np.deg2rad(theta))
        py = dist_hMpc * np.sin(np.deg2rad(theta))

        # Define the length of the box    
        if box_size_hMpc is None:
            x_min, x_max = px.min(), px.max()
            y_min, y_max = py.min(), py.max()
        else:
            x_min, x_max = -box_size_hMpc/2., box_size_hMpc/2.
            y_min, y_max = -box_size_hMpc/2., box_size_hMpc/2.

        # Get the bin of each galaxy
        if nbins is None:
            nbins = int(nGalaxies / np.sqrt(gals_per_bins))

        bins_x = np.linspace(x_min, x_max, nbins+1)
        bins_y = np.linspace(y_min, y_max, nbins+1)
        digit_x = np.digitize(px, bins=bins_x)-1
        digit_y = np.digitize(py, bins=bins_y)-1
    
        px_map, py_map = np.meshgrid((bins_x[:-1]+bins_x[1:])/2.,
        							(bins_y[:-1]+bins_y[1:])/2.,
                                    indexing='ij')
        e1_map = np.zeros((nbins, nbins), dtype=float)
        e2_map = np.zeros((nbins, nbins), dtype=float)
        self.N = np.zeros((nbins, nbins), dtype=int)

        # Average the ellipticities.
        # Should this average be calibrated with the m bias ??
        for ix in range(nbins):
            maskx = digit_x==ix
            for iy in range(nbins):
                masky = digit_y==iy
                mask = maskx*masky
                if mask.sum()==0: continue
                e1_map[iy,ix] = data['e1'][mask].mean() 
                e2_map[iy,ix] = data['e2'][mask].mean()
                self.N[iy,ix] = mask.sum()

        e_mod = np.sqrt(e1_map**2 + e2_map**2)
        beta = np.arctan2(e2_map, e1_map)/2.
        ex_map = e_mod * np.cos(beta)
        ey_map = e_mod * np.sin(beta)

        self.px = px_map
        self.py = py_map
        self.ex = ex_map
        self.ey = ey_map
        self.e1 = e1_map
        self.e2 = e2_map

    def __getitem__(self, key):
        return getattr(self, key)

    def set_Mpc_scale(self, dl):
        Mpc_scale = dl*np.deg2rad(1./3600.)
        self.Mpc_scale_mean = Mpc_scale.mean()
        return Mpc_scale

    def QuickPlot(self, normed=True, cmap=None):
    	if cmap is None:
    		from matplotlib import cm
    		cmap=cm.gist_heat_r

        emod = np.sqrt(self.ex**2+self.ey**2)
        self.quiveropts = dict(headlength=0, headwidth=0, headaxislength=0,
                            width=0.1, pivot='middle', units='xy',
                              alpha=1, color='black')
        plt.figure()
        if normed:
        	plt.quiver(self.px, self.py, self.ex/emod, self.ey/emod, emod, cmap=cmap, **self.quiveropts)
        else:
        	plt.quiver(self.px, self.py, self.ex, self.ey, emod, cmap=cmap, **self.quiveropts)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('$\Vert \gamma \Vert$', fontsize=14)
        plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
        plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
        plt.show()