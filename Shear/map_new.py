import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from joblib import Parallel, delayed

from .. import gentools 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)



def _map_per_bin(mask, dict_per_bin):
    m = dict_per_bin['m'][mask]
    weight = dict_per_bin['weight'][mask]
    e1 = dict_per_bin['e1'][mask]
    e2 = dict_per_bin['e2'][mask]
    sigma_critic = dict_per_bin['sigma_critic'][mask]

    w = weight/sigma_critic**2
    x = {}
    x['accum_w_i'] = np.sum(w, axis=0)
    x['m_cal_num_i'] = np.sum(w*(1+m), axis=0)
    x['shear1_i'] = np.sum(w*e1*sigma_critic, axis=0)
    x['shear2_i'] = np.sum(w*e2*sigma_critic, axis=0)
    x['stat_error_num_i'] = np.sum( (w*0.25*sigma_critic)**2, axis=0)
    x['N_i'] = len(m)
    return x

def _map_per_lens(j, dict_per_lens):
    data_L = dict_per_lens['data_L']
    data_S = dict_per_lens['data_S']
    bins = dict_per_lens['bins']
    back_dz = dict_per_lens['back_dz']
    #cosmo = dict_per_lens['cosmo']

    dL = data_L.iloc[j]
    dS = data_S.loc[dL['CATID']]
    #if back_dz != 0.:
    mask_dz = dS['Z_B'].values >= dL['Z'] + back_dz
    dS = dS[mask_dz]

    DD = gentools.compute_lensing_distances(zl=dL['Z'], zs=dS['Z_B'].values,
        precomputed=True, cache=True)#, cosmo=self.cosmo)

    Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
    sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])
    
    # Compute distance and ellipticity components...
    dist, theta = gentools.sphere_angular_vector(dS['RAJ2000'].values, dS['DECJ2000'].values,
                                                dL['RA'], dL['DEC'], units='deg')
    theta += 90. 
    dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h

    px = dist_hMpc * np.cos(np.deg2rad(theta))
    py = dist_hMpc * np.sin(np.deg2rad(theta))

    digit_x = gentools.digitize(px, bins=bins)
    digit_y = gentools.digitize(py, bins=bins)

    dict_per_bin = {'m': dS['m'].values, 'weight': dS['weight'].values,
                    'digit_x': digit_x, 'digit_y': digit_y, 
                    'sigma_critic': sigma_critic,
                    'e1': dS['e1'].values, 'e2': dS['e2'].values}

    xshape = (len(bins)-1, len(bins)-1)
    x = {'shear1_j': np.zeros(xshape), 'shear2_j': np.zeros(xshape),
         'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
         'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(xshape)}

    # Accumulate the ellipticities.
    for ix in range(nbins):
        maskx = digit_x==ix
        for iy in range(nbins):
            masky = digit_y==iy
            mask = maskx*masky
            if mask.sum()==0: continue
            mp = _map_per_bin(mask, dict_per_bin)
            x['shear1_j'][iy, ix] = mp['shear1_i']
            x['shear2_j'][iy, ix] = mp['shear2_i']
            x['accum_w_j'][iy, ix] = mp['accum_w_i']
            x['m_cal_num_j'][iy, ix] = mp['m_cal_num_i']
            x['stat_error_num_j'][iy, ix] = mp['stat_error_num_i']
            x['N_j'][iy, ix] = mp['N_i']
    return x



class Map(object):

    def __init__(self, nbins=None, box_size_hMpc=None, 
        cosmo=cosmo, back_dz=0., precompute_distances=True, njobs=1):

        self.cosmo = cosmo
        self.njobs = njobs
        self.back_dz = back_dz
        self.nbins = nbins
        self.bins = gentools.make_bins(-box_size_hMpc/2., box_size_hMpc/2., nbins=nbins)

    def __getitem__(self, key):
        return getattr(self, key)

    def QuickPlot(self, normed=True, cmap='gist_heat_r'):

        quiveropts = dict(headlength=0, headwidth=0, headaxislength=0,
                              pivot='middle', units='xy',
                              alpha=1, color='black')
        plt.figure()
        if normed:
            plt.quiver(self.px, self.py, 
                    self.shearx/self.shear, self.sheary/self.shear, 
                    self.shear, cmap=cmap, **quiveropts)
        else:
            plt.quiver(self.px, self.py, 
                    self.shearx, self.sheary, 
                    self.shear, cmap=cmap, **quiveropts)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('$\Vert \gamma \Vert$', fontsize=14)
        plt.xlabel('$r\,[Mpc/h]$', fontsize=14)
        plt.ylabel('$r\,[Mpc/h]$', fontsize=14)
        plt.show()


@gentools.timer
class CompressedMap(Map):

    def __init__(self, data_L, data_S, nbins=None, box_size_hMpc=None, cosmo=cosmo):

        super().__init__(nbins=None, box_size_hMpc=None, cosmo=cosmo)

        if data_S.index.name is not 'CATID':
            data_S_indexed = data_S.set_index('CATID')
        mp = self._map(data_L=data_L, data_S=data_S_indexed)
        mp = self._reduce(mp)
        mp = self._cartesian(mp)

        # Now in units of h*Msun/pc**2
        self.N = mp['N'].astype(np.int32)
        self.beta = mp['beta']
        self.shear = mp['shear']/self.cosmo.h
        self.shearx = mp['shearx']/self.cosmo.h
        self.sheary = mp['sheary']/self.cosmo.h
        self.stat_error = mp['stat_error']/self.cosmo.h

        bins_centre = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.px, self.py = np.meshgrid(bins_centre, bins_centre, indexing='xy')

    def _map(self, data_L, data_S):
        ''' Computes map for CompressedCatalog
        '''
        dict_per_lens = {'data_L': data_L, 'data_S': data_S, 
                        'bins': self.bins, 'back_dz': self.back_dz}

        # Compute maps per lens
        with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
            delayed_fun = delayed(_map_per_lens)
            mp = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
        return mp

    def _reduce(self, mp):
        N = np.sum( [mp[_]['N_j'] for _ in range(len(mp))], axis=0)
        shear1 = np.sum( [mp[_]['shear1_j'] for _ in range(len(mp))], axis=0)
        shear2 = np.sum( [mp[_]['shear2_j'] for _ in range(len(mp))], axis=0)
        accum_w = np.sum( [mp[_]['accum_w_j'] for _ in range(len(mp))], axis=0)
        m_cal_num = np.sum( [mp[_]['m_cal_num_j'] for _ in range(len(mp))], axis=0)
        stat_error_num = np.sum( [mp[_]['stat_error_num_j'] for _ in range(len(mp))], axis=0)

        m_cal = m_cal_num / accum_w
        x = {}
        x['shear1'] = (shear1/accum_w) / m_cal
        x['shear2'] = (shear2/accum_w) / m_cal
        x['stat_error'] = np.sqrt(stat_error_num/accum_w**2) / m_cal
        x['N'] = N
        return x

    def _cartesian(self, mp):
        shear = np.sqrt(mp['shear1']**2 + mp['shear2']**2)
        beta = np.arctan2(mp['shear2'], mp['shear1'])/2.
        x = {}
        x['shear'] = shear
        x['beta'] = beta
        x['shearx'] = shear * np.cos(beta)
        x['sheary'] = shear * np.sin(beta)
        x['stat_error'] = mp['stat_error']
        x['N'] = mp['N']
        return x


class ExpandedMap(Map):

    def __init__(self, data=None, nbins=None, gals_per_bins=50., box_size_hMpc=None, cosmo=cosmo):

        super().__init__(nbins=None, gals_per_bins=50., box_size_hMpc=None, cosmo=cosmo)

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

    def _map(self, data):
        ''' Computes map for CompressedCatalog
        '''
        mask_dz = data['Z_B'] >= data['Z'] + self.back_dz
        data = data[mask_dz]

        DD = gentools.compute_lensing_distances(zl=data['Z'], zs=data['Z_B'],
            precomputed=True, cosmo=self.cosmo)
        
        Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
        sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])

        # Compute distance and ellipticity components...
        dist, theta = gentools.sphere_angular_vector(data['RAJ2000'], data['DECJ2000'],
                                                    data['RA'], data['DEC'], units='deg')
        theta += 90. 
        dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h

        px = dist_hMpc * np.cos(np.deg2rad(theta))
        py = dist_hMpc * np.sin(np.deg2rad(theta))

        digit_x = gentools.digitize(px, bins=self.bins)
        digit_y = gentools.digitize(py, bins=self.bins)

        dict_per_bin = {'m': data['m'].values, 'weight': data['weight'].values,
                        'digit_x': digit_x, 'digit_y': digit_y, 
                        'sigma_critic': sigma_critic,
                        'e1': data['e1'].values, 'e2': data['e2'].values}

        xshape = (len(bins)-1, len(bins)-1)
        x = {'shear1_j': np.zeros(xshape), 'shear2_j': np.zeros(xshape),
             'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
             'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(xshape)}
            return mp

    def _reduce(self, mp):
        N = np.sum( [mp[_]['N_j'] for _ in range(len(mp))], axis=0)
        shear1 = np.sum( [mp[_]['shear1_j'] for _ in range(len(mp))], axis=0)
        shear2 = np.sum( [mp[_]['shear2_j'] for _ in range(len(mp))], axis=0)
        accum_w = np.sum( [mp[_]['accum_w_j'] for _ in range(len(mp))], axis=0)
        m_cal_num = np.sum( [mp[_]['m_cal_num_j'] for _ in range(len(mp))], axis=0)
        stat_error_num = np.sum( [mp[_]['stat_error_num_j'] for _ in range(len(mp))], axis=0)

        m_cal = m_cal_num / accum_w
        x = {}
        x['shear1'] = (shear1/accum_w) / m_cal
        x['shear2'] = (shear2/accum_w) / m_cal
        x['stat_error'] = np.sqrt(stat_error_num/accum_w**2) / m_cal
        x['N'] = N
        return x

    def _cartesian(self, mp):
        shear = np.sqrt(mp['shear1']**2 + mp['shear2']**2)
        beta = np.arctan2(mp['shear2'], mp['shear1'])/2.
        x = {}
        x['shear'] = shear
        x['beta'] = beta
        x['shearx'] = shear * np.cos(beta)
        x['sheary'] = shear * np.sin(beta)
        x['stat_error'] = mp['stat_error']
        x['N'] = mp['N']
        return x



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


