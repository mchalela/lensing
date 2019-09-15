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

    def __init__(self, data=None, nbins=None, gals_per_bins=50., cosmo=cosmo):

        if data is None:
            raise ValueError('ShearMap needs some data to work with...')

        if isinstance(data, pd.DataFrame):
            data = data.to_records()

        # Define some parameters...
        nGalaxies = len(data)
        Mpc_scale = self.set_Mpc_scale(dl=data['DL'])

        # Compute distance and ellipticity components...
        dist, theta = gentools.sphere_angular_vector(data['RAJ2000'], data['DECJ2000'],
                                                    data['RA'], data['DEC'], units='deg')
        #theta += 90. 
        dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # distance to the lens in Mpc/h
        e_mod = np.sqrt(data['e1']**2 + data['e2']**2)

        # Transfrom e1,e2 to cartesian components ex,ey
        cos_theta = np.cos(np.deg2rad(theta))
        sin_theta = np.sin(np.deg2rad(theta))
        px = dist_hMpc * cos_theta
        py = dist_hMpc * sin_theta
        ex = e_mod * cos_theta
        ey = e_mod * sin_theta

        # Get the bin of each galaxy
        if nbins is None:
            nbins = int(nGalaxies / np.sqrt(gals_per_bins))
        self.bins_x = np.linspace(px.min(), px.max(), nbins+1)
        self.bins_y = np.linspace(py.min(), py.max(), nbins+1)
        digit_x = np.digitize(px, bins=self.bins_x)-1
        digit_y = np.digitize(py, bins=self.bins_y)-1
    
        px_map, py_map = np.meshgrid((self.bins_x[:-1]+self.bins_x[1:])/2.,
                                        (self.bins_y[:-1]+self.bins_y[1:])/2.)
        ex_map = np.zeros((nbins, nbins))
        ey_map = np.zeros((nbins, nbins))

        # Average the ellipticities.
        # Should this average be calibrated with the m bias ??
        for ix in range(nbins):
            maskx = digit_x==ix
            for iy in range(nbins):
                masky = digit_y==iy
                mask = maskx*masky
                if mask.sum()==0: continue
                ex_map[iy,ix] = ex[mask].mean() 
                ey_map[iy,ix] = ey[mask].mean()


        quiveropts = dict( headlength=0, headwidth=0, headaxislength=0,
                            linewidth=1, pivot='middle', units='xy',
                              alpha=1, color='black')
        plt.figure()
        plt.quiver(px_map, py_map, ex_map, ey_map,**quiveropts)
        plt.show()
        self.px = px_map
        self.py = py_map
        self.ex = ex_map
        self.ey = ey_map

    def set_Mpc_scale(self, dl):
        Mpc_scale = dl*np.deg2rad(1./3600.)
        self.Mpc_scale_mean = Mpc_scale.mean()
        return Mpc_scale
