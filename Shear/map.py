import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM

from .. import gentools 

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
        sigma_critic = self.set_sigma_critic(dl=data['DL'], ds=data['DS'], dls=data['DLS'])

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
                                    indexing='xy')
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

                weight = data['weight'][mask]/sigma_critic[mask]**2
                m_cal = 1 + np.average(data['m'][mask], weights=weight)

                e1_map[iy, ix] = np.average(data['e1'][mask]*sigma_critic[mask], weights=weight) / m_cal
                e2_map[iy, ix] = np.average(data['e2'][mask]*sigma_critic[mask], weights=weight) / m_cal

                self.N[iy, ix] = mask.sum()

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

    def _mirror(self, data, mirror):

        ra_gal, dec_gal = cat.data['RAJ2000'], cat.data['DECJ2000'] 
        ra_cen, dec_cen = cat.data['RA'], cat.data['DEC']
        e1_gal, e2_gal = cat.data['e1'], cat.data['e2']

        angular_vector = gentools.sphere_angular_vector
        rotate_pos = gentools.equatorial_coordinates_rotation
        rotate_ellip = gentools.polar_rotation        

        distance, orientation = angular_vector(
            ra_gal, dec_gal, ra_cen, dec_cen, units='deg')
        orientation += 90.

        # Third quadrant
        if miror in ['x', 'xy', 'yx']:
            mask_3c = (orientation>180.)*(orientation<270)
            ang = orientation[mask_3c] - 180
            ra_rot_3c, dec_rot_3c = rotate_pos(
                ra_gal[mask_3c], dec_gal[mask_3c],
                ra_cen[mask_3c], dec_cen[mask_3c],
                -2*ang, units='deg')

        # Fourth quadrant
        if miror in ['y', 'xy', 'yx']:
            mask_4c = (orientation>=270.)*(orientation<360)
            ang = 360 - orientation[mask_4c]
            ra_rot_4c, dec_rot_4c = rotate_pos(
                ra_gal[mask_4c], dec_gal[mask_4c],
                ra_cen[mask_4c], dec_cen[mask_4c],
                2*ang, units='deg')





    def set_Mpc_scale(self, dl):
        Mpc_scale = dl*np.deg2rad(1./3600.)
        self.Mpc_scale_mean = Mpc_scale.mean()
        return Mpc_scale

    def set_sigma_critic(self, dl, ds, dls):
        beta = dls/ds
        sigma_critic = cvel**2/(4.*np.pi*G*(dl*1e6*pc)) * (1./beta) * (pc**2/Msun)
        self.beta_mean = beta.mean()
        self.sigma_critic = sigma_critic.mean()
        return sigma_critic


    def QuickPlot(self, normed=True, cmap='gist_heat_r'):

        emod = np.sqrt(self.ex**2+self.ey**2)
        self.quiveropts = dict(headlength=0, headwidth=0, headaxislength=0,
                              pivot='middle', units='xy',
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