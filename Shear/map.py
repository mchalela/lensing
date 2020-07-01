import os, platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
#from matplotlib.colors import LogNorm
from astropy.cosmology import LambdaCDM
from joblib import Parallel, delayed

from .. import gentools 

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)

# PICKABLE functions =================================================================
def _deltasigma_map_per_bin(mask, dict_per_bin):
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
    x['sigma_critic_i'] = np.sum(w*sigma_critic, axis=0)
    x['N_i'] = len(m)
    return x

def _shearmap_per_bin(mask, dict_per_bin):
    m = dict_per_bin['m'][mask]
    weight = dict_per_bin['weight'][mask]
    e1 = dict_per_bin['e1'][mask]
    e2 = dict_per_bin['e2'][mask]
    sigma_critic = dict_per_bin['sigma_critic'][mask]

    w = weight/sigma_critic**2
    x = {}
    x['accum_w_i'] = np.sum(w, axis=0)
    x['m_cal_num_i'] = np.sum(w*(1+m), axis=0)
    x['shear1_i'] = np.sum(w*e1, axis=0)
    x['shear2_i'] = np.sum(w*e2, axis=0)
    x['stat_error_num_i'] = np.sum( (w*0.25)**2, axis=0)
    x['sigma_critic_i'] = np.sum(w*sigma_critic, axis=0)
    x['N_i'] = len(m)
    return x

def _map_per_lens(j, dict_per_lens):
    #print(j)
    data_L = dict_per_lens['data_L']
    data_S = dict_per_lens['data_S']
    scale = dict_per_lens['scale']
    bins = dict_per_lens['bins']
    back_dz = dict_per_lens['back_dz']
    rotate = dict_per_lens['rotate']
    dN = dict_per_lens['colnames']
    map_flag = dict_per_lens['map_flag']
    nbins = len(bins)-1
    #cosmo = dict_per_lens['cosmo']

    dL = data_L.iloc[j]
    try:
        dS = data_S.loc[dL['CATID']]
    except Exception as e:
        dS = data_S.reindex(dL['CATID']).dropna()
    #if back_dz != 0.:
    mask_dz = dS['Z_B'].values >= dL[dN['Z']] + back_dz
    dS = dS[mask_dz]

    DD = gentools.compute_lensing_distances(zl=dL[dN['Z']], zs=dS['Z_B'].values,
        precomputed=True, cache=True)#, cosmo=self.cosmo)

    Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
    sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])
    
    # Compute distance and ellipticity components...
    dist, theta = gentools.sphere_angular_vector(dS['RAJ2000'].values, dS['DECJ2000'].values,
                                                dL[dN['RA']], dL[dN['DEC']], units='deg')
    dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h
    if scale is not None:
        dist_hMpc /= dL[scale]

    # Rotation, if needed
    if rotate is not None:
        rot_angle = 90 - dL[rotate] #+ 90.
        theta = 90 - theta - rot_angle # align to the X axis
        e1, e2 = gentools.polar_rotation(dS['e1'].values, -dS['e2'].values, theta=-np.deg2rad(rot_angle))
    else:
        theta = 90. - theta
        e1, e2 = dS['e1'].values, -dS['e2'].values

    px = dist_hMpc * np.cos(np.deg2rad(theta))
    py = dist_hMpc * np.sin(np.deg2rad(theta))

    digit_x = gentools.digitize(px, bins=bins)
    digit_y = gentools.digitize(py, bins=bins)

    dict_per_bin = {'m': dS['m'].values, 'weight': dS['weight'].values,
                    'digit_x': digit_x, 'digit_y': digit_y, 
                    'sigma_critic': sigma_critic,
                    'e1': e1, 'e2': e2}

    xshape = (nbins, nbins)
    x = {'shear1_j': np.zeros(xshape), 'shear2_j': np.zeros(xshape),
         'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
         'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(xshape),
         'sigma_critic_j': np.zeros(xshape)}

    # Accumulate the ellipticities.
    for ix in range(nbins):
        maskx = digit_x==ix
        for iy in range(nbins):
            masky = digit_y==iy
            mask = maskx*masky
            if mask.sum()==0: continue
            if map_flag == 'deltasigma':
                mp = _deltasigma_map_per_bin(mask, dict_per_bin)
            else:
                mp = _shearmap_per_bin(mask, dict_per_bin)
            x['shear1_j'][iy, ix] = mp['shear1_i']
            x['shear2_j'][iy, ix] = mp['shear2_i']
            x['accum_w_j'][iy, ix] = mp['accum_w_i']
            x['m_cal_num_j'][iy, ix] = mp['m_cal_num_i']
            x['stat_error_num_j'][iy, ix] = mp['stat_error_num_i']
            x['sigma_critic_j'][iy, ix] = mp['sigma_critic_i']
            x['N_j'][iy, ix] = mp['N_i']
    return x

# ============================================================================================
def read_map(file):
    ''' Read profile written with Map.write_to() method
    '''
    from astropy.io import fits

    with fits.open(file) as f:
        mp = Map()
        mp.px = f['px'].data
        mp.py = f['py'].data
        mp.shear = f['shear'].data
        mp.beta = f['beta'].data
        mp.stat_error = f['stat_error'].data
        mp.N = f['N'].data
        # Complete variables
        mp.shearx = mp.shear * np.cos(mp.beta)
        mp.sheary = mp.shear * np.sin(mp.beta)
        mp.shear1 = mp.shear * np.cos(2*mp.beta)
        mp.shear2 = mp.shear * np.sin(2*mp.beta)
    return mp


class Map(object):

    def __init__(self, nbins=None, box_size=None, cosmo=None, back_dz=None):

        self.cosmo = cosmo
        self.back_dz = back_dz
        self.nbins = nbins
        self.box_size = box_size
        if box_size is not None:
            self.bins = gentools.make_bins(-box_size/2., box_size/2., nbins=nbins, space='lin') 

        self.px, self.py = None, None
        self.shearx, self.sheary = None, None
        self.shear1, self.shear2 = None, None
        self.shear, self.beta = None, None
        self.stat_error = None
        self.N = None

    def __getitem__(self, key):
        return getattr(self, key)

    def write_to(self, file, header=None, overwrite=False):
        '''Write Map to a FITS file.
        Add a header to lensing.shear.Profile output file
        to know the sample parameters used to build it.
        
         file:      (str) Name of output file
         header:    (dic) Dictionary with parameter cuts. Optional.
                Example: {'z_min':0.1, 'z_max':0.3, 'odds_min':0.5}
         overwrite: (bool) Flag to overwrite file if it already exists.
        '''
        from astropy.io import fits
        
        if os.path.isfile(file):
            if overwrite:
                os.remove(file)
            else:
                raise IOError('File already exist. You may want to use overwrite=True.')
        
        hdulist = [fits.PrimaryHDU()]
        for atr in ['px', 'py', 'shear', 'beta', 'stat_error', 'N']:
            hdulist.append(fits.ImageHDU(self.__getitem__(atr), name=atr))
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(file, overwrite=overwrite)
        return None


@gentools.timer
class SigmaMap(Map):

    def __init__(self, data_L, data_S, scale=None, mirror=None, nbins=10, box_size=0.5, rotate=None,
        cosmo=cosmo, back_dz=0.1, precomputed_distances=True, njobs=1, colnames=None):

        super().__init__(nbins=nbins, box_size=box_size, cosmo=cosmo, back_dz=back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.rotate = rotate
        self.scale = scale

        if colnames is None: colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}
        self.colnames = colnames

        if data_S.index.name is not 'CATID':
            data_S_indexed = data_S.set_index('CATID')

        mp = self._map(data_L=data_L, data_S=data_S_indexed)
        mp = self._accum(mp)
        if mirror is not None:
            mp = self._mirror(mp, mirror=mirror)
        mp = self._reduce(mp)
        mp = self._cartesian(mp)

        # Now in units of h*Msun/pc**2
        self.sigma_critic = mp['sigma_critic']/self.cosmo.h
        self.N = mp['N'].astype(np.int32)
        self.beta = mp['beta']
        self.shear  = mp['shear']/self.cosmo.h
        self.shear1 = mp['shear1']/self.cosmo.h
        self.shear2 = mp['shear2']/self.cosmo.h
        self.shearx = mp['shearx']/self.cosmo.h
        self.sheary = mp['sheary']/self.cosmo.h
        self.stat_error = mp['stat_error']/self.cosmo.h

        bins_centre = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.px, self.py = np.meshgrid(bins_centre, bins_centre, indexing='xy')

    def _map(self, data_L, data_S):
        ''' Computes map for CompressedCatalog
        '''
        dict_per_lens = {'data_L': data_L, 'data_S': data_S, 'scale': self.scale,
                        'bins': self.bins, 'back_dz': self.back_dz,
                        'rotate': self.rotate, 'colnames': self.colnames,
                        'map_flag': 'deltasigma'}

        # Compute maps per lens
        with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
            delayed_fun = delayed(_map_per_lens)
            mp = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
        return mp

    def _accum(self, mp):
        x = {}      
        x['N'] = np.sum( [mp[_]['N_j'] for _ in range(len(mp))], axis=0)
        x['shear1'] = np.sum( [mp[_]['shear1_j'] for _ in range(len(mp))], axis=0)
        x['shear2'] = np.sum( [mp[_]['shear2_j'] for _ in range(len(mp))], axis=0)
        x['accum_w'] = np.sum( [mp[_]['accum_w_j'] for _ in range(len(mp))], axis=0)
        x['m_cal_num'] = np.sum( [mp[_]['m_cal_num_j'] for _ in range(len(mp))], axis=0)
        x['stat_error_num'] = np.sum( [mp[_]['stat_error_num_j'] for _ in range(len(mp))], axis=0)
        x['sigma_critic'] = np.sum( [mp[_]['sigma_critic_j'] for _ in range(len(mp))], axis=0)
        return x

    def _mirror(self, mp, mirror):
        if mirror.lower() == 'x': axis = [1]
        if mirror.lower() == 'y': axis = [0]
        if mirror.lower() == 'xy': axis = [1, 0]
        if mirror.lower() == 'yx': axis = [0, 1]

        for ax in axis:
            mp['shear2'] += -1*np.flip(mp['shear2'], axis=ax)   # only shear2 changes sign in each mirroring
            for attr in ['shear1', 'N', 'accum_w', 'm_cal_num', 'stat_error_num', 'sigma_critic']:
                mp[attr] += np.flip(mp[attr], axis=ax)
        return mp

    def _reduce(self, mp):
        x = {}
        m_cal = mp['m_cal_num'] / mp['accum_w']
        x['shear1'] = (mp['shear1']/mp['accum_w']) / m_cal
        x['shear2'] = (mp['shear2']/mp['accum_w']) / m_cal
        x['stat_error'] = np.sqrt(mp['stat_error_num']/mp['accum_w']**2) / m_cal
        x['sigma_critic'] = mp['sigma_critic'].sum()/mp['accum_w'].sum()
        x['N'] = mp['N']
        return x

    def _cartesian(self, mp):
        shear = np.sqrt(mp['shear1']**2 + mp['shear2']**2)
        beta = np.arctan2(mp['shear2'], mp['shear1'])/2.
        mp['shearx'] = shear * np.cos(beta)
        mp['sheary'] = shear * np.sin(beta)
        mp['shear'] = shear
        mp['beta'] = beta
        return mp

    def QuickPlot(self, normed=True, cmap='gist_heat_r'):

        #norm = LogNorm(vmin=1., vmax=self.shear.max(), clip=True)
        quiveropts = dict(headlength=0, headwidth=0, headaxislength=0,
                              pivot='middle', units='xy',
                              alpha=1, color='black')#, norm=norm)
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
        cbar.ax.set_ylabel('$\Delta\Sigma [h\,M_{\odot}/pc^2]$', fontsize=12)
        plt.xlabel('$r\,[Mpc/h]$', fontsize=12)
        plt.ylabel('$r\,[Mpc/h]$', fontsize=12)
        plt.show()
        return None


@gentools.timer
class ShearMap(Map):

    def __init__(self, data_L, data_S, scale=None, mirror=None, nbins=10, box_size=0.5, rotate=None,
        cosmo=cosmo, back_dz=0.1, precomputed_distances=True, njobs=1, colnames=None):

        super().__init__(nbins=nbins, box_size=box_size, cosmo=cosmo, back_dz=back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.rotate = rotate
        self.scale = scale

        if colnames is None: colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}
        self.colnames = colnames

        if data_S.index.name is not 'CATID':
            data_S_indexed = data_S.set_index('CATID')

        mp = self._map(data_L=data_L, data_S=data_S_indexed)
        mp = self._accum(mp)
        if mirror is not None:
            mp = self._mirror(mp, mirror=mirror)
        mp = self._reduce(mp)
        mp = self._cartesian(mp)

        self.N = mp['N'].astype(np.int32)
        self.beta = mp['beta']
        self.shear  = mp['shear']
        self.shear1 = mp['shear1']
        self.shear2 = mp['shear2']
        self.shearx = mp['shearx']
        self.sheary = mp['sheary']
        self.stat_error = mp['stat_error']
        # Now in units of h*Msun/pc**2
        self.sigma_critic = mp['sigma_critic']/self.cosmo.h

        bins_centre = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.px, self.py = np.meshgrid(bins_centre, bins_centre, indexing='xy')

    def _map(self, data_L, data_S):
        ''' Computes map for CompressedCatalog
        '''
        dict_per_lens = {'data_L': data_L, 'data_S': data_S, 'scale': self.scale,
                        'bins': self.bins, 'back_dz': self.back_dz,
                        'rotate': self.rotate, 'colnames': self.colnames,
                        'map_flag': 'shear'}

        # Compute maps per lens
        with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
            delayed_fun = delayed(_map_per_lens)
            mp = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
        return mp

    def _accum(self, mp):
        x = {}      
        x['N'] = np.sum( [mp[_]['N_j'] for _ in range(len(mp))], axis=0)
        x['shear1'] = np.sum( [mp[_]['shear1_j'] for _ in range(len(mp))], axis=0)
        x['shear2'] = np.sum( [mp[_]['shear2_j'] for _ in range(len(mp))], axis=0)
        x['accum_w'] = np.sum( [mp[_]['accum_w_j'] for _ in range(len(mp))], axis=0)
        x['m_cal_num'] = np.sum( [mp[_]['m_cal_num_j'] for _ in range(len(mp))], axis=0)
        x['stat_error_num'] = np.sum( [mp[_]['stat_error_num_j'] for _ in range(len(mp))], axis=0)
        x['sigma_critic'] = np.sum( [mp[_]['sigma_critic_j'] for _ in range(len(mp))], axis=0)
        return x

    def _mirror(self, mp, mirror):
        if mirror.lower() == 'x': axis = [1]
        if mirror.lower() == 'y': axis = [0]
        if mirror.lower() == 'xy': axis = [1, 0]
        if mirror.lower() == 'yx': axis = [0, 1]

        for ax in axis:
            mp['shear2'] += -1*np.flip(mp['shear2'], axis=ax)   # only shear2 changes sign in each mirroring
            for attr in ['shear1', 'N', 'accum_w', 'm_cal_num', 'stat_error_num', 'sigma_critic']:
                mp[attr] += np.flip(mp[attr], axis=ax)
        return mp

    def _reduce(self, mp):
        x = {}
        m_cal = mp['m_cal_num'] / mp['accum_w']
        x['shear1'] = (mp['shear1']/mp['accum_w']) / m_cal
        x['shear2'] = (mp['shear2']/mp['accum_w']) / m_cal
        x['stat_error'] = np.sqrt(mp['stat_error_num']/mp['accum_w']**2) / m_cal
        x['sigma_critic'] = mp['sigma_critic'].sum()/mp['accum_w'].sum()
        x['N'] = mp['N']
        return x

    def _cartesian(self, mp):
        shear = np.sqrt(mp['shear1']**2 + mp['shear2']**2)
        beta = np.arctan2(mp['shear2'], mp['shear1'])/2.
        mp['shearx'] = shear * np.cos(beta)
        mp['sheary'] = shear * np.sin(beta)
        mp['shear'] = shear
        mp['beta'] = beta
        return mp

    def QuickPlot(self, normed=True, cmap='gist_heat_r'):

        #norm = LogNorm(vmin=1., vmax=self.shear.max(), clip=True)
        quiveropts = dict(headlength=0, headwidth=0, headaxislength=0,
                              pivot='middle', units='xy',
                              alpha=1, color='black')#, norm=norm)
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
        cbar.ax.set_ylabel('$\gamma$', fontsize=12)
        plt.xlabel('$r\,[Mpc/h]$', fontsize=12)
        plt.ylabel('$r\,[Mpc/h]$', fontsize=12)
        plt.show()
        return None


#####################################################################################################
# Deprecated

@gentools.timer
class ExpandedMap(Map):
    def __init__(self, data=None, nbins=10, box_size_hMpc=0.5, 
        cosmo=cosmo, back_dz=0.1, precomputed_distances=True, njobs=1):

        super().__init__(nbins=nbins, box_size_hMpc=box_size_hMpc, cosmo=cosmo, back_dz=back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances

        mp = self._map(data=data)
        mp = self._reduce(mp)
        mp = self._cartesian(mp)

        # Now in units of h*Msun/pc**2
        self.N = mp['N'].astype(np.int32)
        self.beta = mp['beta']
        self.shear  = mp['shear']/self.cosmo.h
        self.shear1 = mp['shear1']/self.cosmo.h
        self.shear2 = mp['shear2']/self.cosmo.h
        self.shearx = mp['shearx']/self.cosmo.h
        self.sheary = mp['sheary']/self.cosmo.h
        self.stat_error = mp['stat_error']/self.cosmo.h
        
        bins_centre = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.px, self.py = np.meshgrid(bins_centre, bins_centre, indexing='xy')

    def _map(self, data):
        ''' Computes map for CompressedCatalog
        '''
        mask_dz = data['Z_B'].values >= data['Z'].values + self.back_dz
        data = data[mask_dz]

        DD = gentools.compute_lensing_distances(zl=data['Z'].values, zs=data['Z_B'].values,
            precomputed=self.precomputed_distances, cache=True)#, cosmo=self.cosmo)

        Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
        sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])
        
        # Compute distance and ellipticity components...
        dist, theta = gentools.sphere_angular_vector(data['RAJ2000'].values, data['DECJ2000'].values,
                                                    data['RA'].values, data['DEC'].values, units='deg')
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

        xshape = (self.nbins, self.nbins)
        x = {'shear1_j': np.zeros(xshape), 'shear2_j': np.zeros(xshape),
             'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
             'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(xshape)}

        with Parallel(n_jobs=self.njobs) as parallel:
            delayed_fun = delayed(_map_per_bin)
            mp = parallel(delayed_fun(
                (digit_x==ix)*(digit_y==iy), dict_per_bin
                ) for ix in range(self.nbins) for iy in range(self.nbins))

        return mp

    def _reduce(self, mp):
        N = np.array( [mp[_]['N_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T
        shear1 = np.array( [mp[_]['shear1_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T
        shear2 = np.array( [mp[_]['shear2_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T
        accum_w = np.array( [mp[_]['accum_w_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T
        m_cal_num = np.array( [mp[_]['m_cal_num_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T
        stat_error_num = np.array( [mp[_]['stat_error_num_i'] for _ in range(len(mp))]).reshape((self.nbins, self.nbins)).T

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
        mp['shearx'] = shear * np.cos(beta)
        mp['sheary'] = shear * np.sin(beta)
        mp['shear'] = shear
        mp['beta'] = beta
        return mp
