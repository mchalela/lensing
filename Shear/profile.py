import os, platform
import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from joblib import Parallel, delayed

from .. import gentools

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
cvel = 299792458.   # Speed of light (m.s-1)
G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16  # 1 pc (m)
Msun = 1.989e30     # Solar mass (kg)

np.random.seed(0)

# PICKABLE functions =================================================================
def _deltasigma_profile_per_bin(i, dict_per_bin):

    nboot = dict_per_bin['nboot']

    #mask = dict_per_bin['digit']==i
    #N_i = mask.sum()
    mask = np.where(dict_per_bin['digit']==i)[0]
    N_i = len(mask)
    if N_i==0: 
        x = {'shear_i': np.zeros(nboot+1), 'cero_i': np.zeros(nboot+1),
             'accum_w_i': np.zeros(nboot+1), 'm_cal_num_i': np.zeros(nboot+1),
             'stat_error_num_i': np.zeros(nboot+1), 'N_i': 0,
             'sigma_critic_i': np.zeros(nboot+1)}
        return x

    m = dict_per_bin['m'][mask]
    weight = dict_per_bin['weight'][mask]
    et = dict_per_bin['et'][mask]
    ex = dict_per_bin['ex'][mask]
    sigma_critic = dict_per_bin['sigma_critic'][mask]
    # Create boot sample
    if nboot>0:
        krand = np.random.choice(np.arange(N_i), (N_i, nboot), replace=True)
        m = np.hstack((m[:, np.newaxis], m[krand]))
        weight = np.hstack((weight[:, np.newaxis], weight[krand]))
        et = np.hstack((et[:, np.newaxis], et[krand]))
        ex = np.hstack((ex[:, np.newaxis], ex[krand]))
        sigma_critic = np.hstack((sigma_critic[:, np.newaxis], sigma_critic[krand]))
    
    w = weight/sigma_critic**2
    x = {}
    x['accum_w_i'] = np.sum(w, axis=0)
    x['sigma_critic_i'] = np.sum(sigma_critic, axis=0)
    x['m_cal_num_i'] = np.sum(w*(1+m), axis=0)
    x['shear_i'] = np.sum(w*et*sigma_critic, axis=0)
    x['cero_i'] = np.sum(w*ex*sigma_critic, axis=0)
    x['stat_error_num_i'] = np.sum( (w*0.25*sigma_critic)**2, axis=0)
    if nboot==0:
        for k in x.keys():
            if isinstance(x[k], np.ndarray): break
            x[k] = np.array([x[k]])

    x['N_i'] = N_i
    return x


def _shear_profile_per_bin(i, dict_per_bin):

    nboot = dict_per_bin['nboot']

    #mask = dict_per_bin['digit']==i
    #N_i = mask.sum()
    mask = np.where(dict_per_bin['digit']==i)[0]
    N_i = len(mask)
    if N_i==0: 
        x = {'shear_i': np.zeros(nboot+1), 'cero_i': np.zeros(nboot+1),
             'accum_w_i': np.zeros(nboot+1), 'm_cal_num_i': np.zeros(nboot+1),
             'stat_error_num_i': np.zeros(nboot+1), 'N_i': 0,
             'sigma_critic_i': np.zeros(nboot+1)}
        return x

    m = dict_per_bin['m'][mask]
    weight = dict_per_bin['weight'][mask]
    et = dict_per_bin['et'][mask]
    ex = dict_per_bin['ex'][mask]
    sigma_critic = dict_per_bin['sigma_critic'][mask]
    # Create boot sample
    if nboot>0:
        krand = np.random.choice(np.arange(N_i), (N_i, nboot), replace=True)
        m = np.hstack((m[:, np.newaxis], m[krand]))
        weight = np.hstack((weight[:, np.newaxis], weight[krand]))
        et = np.hstack((et[:, np.newaxis], et[krand]))
        ex = np.hstack((ex[:, np.newaxis], ex[krand]))
        sigma_critic = np.hstack((sigma_critic[:, np.newaxis], sigma_critic[krand]))
    
    w = weight/sigma_critic**2
    x = {}
    x['accum_w_i'] = np.sum(w, axis=0)
    x['sigma_critic_i'] = np.sum(sigma_critic, axis=0)
    x['m_cal_num_i'] = np.sum(w*(1+m), axis=0)
    x['shear_i'] = np.sum(w*et, axis=0)
    x['cero_i'] = np.sum(w*ex, axis=0)
    x['stat_error_num_i'] = np.sum( (w*0.25)**2, axis=0)
    if nboot==0:
        for k in x.keys():
            if isinstance(x[k], np.ndarray): break
            x[k] = np.array([x[k]])

    x['N_i'] = N_i
    return x


def _profile_per_lens(j, dict_per_lens):
    data_L = dict_per_lens['data_L']
    data_S = dict_per_lens['data_S']
    scale = dict_per_lens['scale']
    bins = dict_per_lens['bins']
    back_dz = dict_per_lens['back_dz']
    nboot = dict_per_lens['nboot']
    precomp_dist = dict_per_lens['precomputed_distances']
    dN = dict_per_lens['colnames']
    pf_flag = dict_per_lens['pf_flag']
    cosmo = dict_per_lens['cosmo']
    lcoord = dict_per_lens['lcoord']

    dL = data_L.iloc[j]
    try:
        dS = data_S.loc[dL['CATID']]
    except Exception as e:
        dS = data_S.reindex(dL['CATID']).dropna()
    #if back_dz != 0.:
    mask_dz = dS['Z_B'].values >= dL[dN['Z']] + back_dz
    dS = dS[mask_dz]

    # gentools.compute_lensing_distances() returns DANG distances
    DD = gentools.compute_lensing_distances(zl=dL[dN['Z']], zs=dS['Z_B'].values,
        precomputed=precomp_dist, cache=True, cosmo=cosmo)
    if lcoord=='comov':
        z_factor = 1. + dL[dN['Z']]
    else:
        z_factor = 1.

    Mpc_scale = gentools.Mpc_scale(dl=DD['DL']) * z_factor
    sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS']) / z_factor**2
    

    # Compute distance and ellipticity components...
    dist, theta = gentools.sphere_angular_vector(dS['RAJ2000'].values, dS['DECJ2000'].values,
                                                    dL[dN['RA']], dL[dN['DEC']], units='deg')
    #theta += 90. 
    dist_Mpc = np.tan(np.deg2rad(dist)) * DD['DL'] #* z_factor   # radial distance to the lens centre in Mpc
    #dist_Mpc = dist*3600. * Mpc_scale   # radial distance to the lens centre in Mpc
    neg_et, ex = gentools.polar_rotation(dS['e1'].values, -dS['e2'].values, -np.deg2rad(90-theta))
    et = -neg_et

    if scale is not None:
        dist_Mpc /= dL[scale]

    digit = np.digitize(dist_Mpc, bins=bins)-1
        
    dict_per_bin = {'m': dS['m'].values, 
                    'weight': dS['weight'].values,
                    'digit': digit, 
                    'sigma_critic': sigma_critic,
                    'et': et, 'ex': ex, 
                    'nboot': nboot}

    xshape = (nboot+1, len(bins)-1)
    x = {'shear_j': np.zeros(xshape), 'cero_j': np.zeros(xshape),
         'accum_w_j': np.zeros(xshape), 'm_cal_num_j': np.zeros(xshape),
         'stat_error_num_j': np.zeros(xshape), 'N_j': np.zeros(len(bins)-1),
         'sigma_critic_j': np.zeros(xshape)}

    for i in range(len(bins)-1):
        if pf_flag == 'deltasigma':
            pf = _deltasigma_profile_per_bin(i, dict_per_bin)
        else:
            pf = _shear_profile_per_bin(i, dict_per_bin)       
        x['shear_j'][:, i] = pf['shear_i']
        x['cero_j'][:, i] = pf['cero_i']
        x['accum_w_j'][:, i] = pf['accum_w_i']
        x['sigma_critic_j'][:, i] = pf['sigma_critic_i']
        x['m_cal_num_j'][:, i] = pf['m_cal_num_i']
        x['stat_error_num_j'][:, i] = pf['stat_error_num_i']
        x['N_j'][i] = pf['N_i']

    return x
# =============================================================================================

def read_profile(file, colnames=True):
    ''' Read profile written with Profile.write_to() method
    '''
    with open(file,'r') as f:
        for i, line in enumerate(f):
            if not line.startswith('#'): break
        f.seek(0,0)
        pf = np.genfromtxt(f, dtype=None, names=colnames, skip_header=i)

    p = Profile()
    p.r = pf['r']
    p.shear = pf['shear']
    p.cero = pf['cero']
    p.shear_error = pf['shear_error']
    p.cero_error = pf['cero_error']
    p.stat_error = pf['stat_error']
    p.N = pf['N']
    return p

class Profile(object):

    def __init__(self, rin=0.1, rout=10., nbins=10, space='log',
        nboot=0, cosmo=cosmo, back_dz=0.):
        
        #if not isinstance(cat, ExpandedCatalog):
        #   raise TypeError('cat must be a LensCat.ExpandedCatalog catalog.')

        self.cosmo = cosmo
        self.nboot = nboot
        self.back_dz = back_dz

        self.bins = gentools.make_bins(rin, rout, nbins=nbins, space=space)
        self.nbins = nbins
        self.rin = self.bins[0]
        self.rout = self.bins[-1]

        # Initialize
        self.shear = None
        self.cero = None
        self.shear_error = None
        self.cero_error = None
        self.stat_error = None
        self.r = None
        self.N = None
        self.nlens = None
        self.cov = None


    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        try:
            self.__str
        except AttributeError as e:
            p = np.column_stack((self.r, self.shear, self.shear_error,
                             self.cero, self.cero_error, self.stat_error, self.N))
            pdf = pd.DataFrame(p, columns=['r','shear','shear_error','cero','cero_error','stat_error','N'])
            self.__str = str(pdf)
        finally:
            return self.__str

    def __repr__(self):
        return str(self)

    def _write_to(self, file, header=None, colnames=True, overwrite=False):
        ''' DEPRECATED
        Add a header to lensing.shear.Profile output file
        to know the sample parameters used to build it.
        
         file:      (str) Name of output file
         header:    (dic) Dictionary with parameter cuts. Optional.
                Example: {'z_min':0.1, 'z_max':0.3, 'odds_min':0.5}
         colnames:  (bool) Flag to write columns names as first line.
                If False, column names are commented.
        '''
        if os.path.isfile(file):
            if overwrite:
                os.remove(file)
            else:
                raise IOError('File already exist. You may want to use overwrite=True.')

        with open(file, 'a') as f:
            f.write('# '+'-'*48+'\n')
            f.write('# '+'\n')
            f.write('# Lensing profile '+'\n')
            if header is not None:
                for key, value in list(header.items()):
                    f.write('# '+key.ljust(14)+' = '+str(value) +'\n')

            f.write('# '+'\n')
            f.write('# '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'\n')
            f.write('# '+'-'*48+'\n')

            C = ' ' if colnames else '# '
            f.write(C+'r  shear  shear_error  cero  cero_error  stat_error  N \n')
            p = np.column_stack((self.r, self.shear, self.shear_error,
                                 self.cero, self.cero_error, self.stat_error, self.N))
            np.savetxt(f, p, fmt=['%12.6f']*6+['%8i'])      
        return None

    def write_to(self, file, header=None, colnames=True, overwrite=False):
        '''Add a header to lensing.shear.Profile output file
        to know the sample parameters used to build it.
        
         file:      (str) Name of output file
         header:    (dic) Dictionary with parameter cuts. Optional.
                Example: {'z_min':0.1, 'z_max':0.3, 'odds_min':0.5}
         colnames:  (bool) Flag to write columns names as first line.
                If False, column names are commented.
        '''
        if os.path.isfile(file):
            if overwrite:
                os.remove(file)
            else:
                raise IOError('File already exist. You may want to use overwrite=True.')

        names = ['r', 'shear', 'cero', 'stat_error', 'N']
        data_arr = np.column_stack((self.r, self.shear, self.cero, self.stat_error, self.N))
        data = pd.DataFrame(data_arr, columns=names)

        # EXTENSION 0 =================================
        hdulist = [fits.PrimaryHDU()]
        
        # EXTENSION 1 =================================
        cols = []
        for name, arr in data.iteritems():
            fmt = '1K' if name == 'N' else '1D'
            c = fits.Column(name=name, format=fmt, array=arr.to_numpy())
            cols.append(c)
        hdu = fits.BinTableHDU.from_columns(cols)
        
        # reorder format cards
        for line in list(hdu.header.items()):
            if 'TFORM' in line[0]:
                del hdu.header[line[0]]
                hdu.header.append(line, end=True)
        
        # user metadata
        if header is not None:
            hdu.header.append(('METADATA', '==========='))
            for key, value in list(header.items()):
                hdu.header.append((str(key), str(value)))
        hdulist.append(hdu)

        # EXTENSION 2 =================================
        hdu_cov = fits.ImageHDU(self.cov)
        hdulist.append(hdu_cov)

        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(file, overwrite=overwrite)   
        return None


@gentools.timer
class DeltaSigmaProfile(Profile):
    '''Profile Constructor: DeltaSigma Profile
    Builds the radial density contrast profile using the lens and source catalog
    constructed with LensCat. Only works with the *compressed* catalog.

    Parameters
    ----------
    data_L : pandas dataframe
        Lens data attribute of LensCat.CompressedCatalog catalog, i.e: cat.data_L 
    data_S : pandas dataframe
        Source data attribute of LensCat.CompressedCatalog catalog, i.e: cat.data_S
    rin : float
        Inner radius to compute the binning. Units of Mpc. If scale parameter is
        specified the value should be in units (a factor) of scale. Default: 0.1.
    rout : float
        Outter radius to compute the binning. Units of Mpc. If scale parameter is
        specified the value should be in units (a factor) of scale. Default: 10.
    nbins : int
        Number of bins to compute the profile. Default: 10.
    scale : str
        Name of the column in data_L that will be used to scale the radial distances
        to the source galaxies. Should be in units of Mpc. Default: None.
    space : str
        String indicating if the radial binning should be logarithmic or linear.
        Possible values are: 'log' and 'lin'. Default: 'log'.
    nboot : int
        Number of bootstrap realizations to compute the error bars. Default: 0.
    cosmo : object
        Object indicating the cosmology. Must be instance an of astropy.cosmology.LambdaCDM
        Default: LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).
    back_dz : float
        Redshift gap between the lens and its sources. This will keep those galaxies
        with Zsources > Zlens + back_dz. Default: 0.
    precomputed_distances : bool
        Flag indicating if the lensing distances should be computed exactly or
        interpolated from a precomputed file.
    njobs : int
        Number of cores for parallelization. Default: 1.
    colnames : dict
        The lens column names for RA, DEC and Z in data_L are expected to be RA, DEC and Z.
        If that is not the case you can pass a dict with the new names.
        Example: colnames = {'RA': 'differentRAname', 'DEC': 'differentDECname', 'Z': 'Z'}.
        Default: None. This means colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}.
    lensing_coord : str
        Lensing coordinates, can be either 'DANG' or 'COMOV'.

    Attributes
    ----------
    r : array, float 
        Center of radial bins. If not scaled, in units of Mpc/h.
    shear : array, float
        Tangential Delta Sigma value for each bin. In units of Msun/pc**2.
    shear_error : array, float
        Bootstrap error in tangential Delta Sigma value for each bin. In units of Msun/pc**2.
        Computed only if nboot>0.
    cero : array, float
        Corssed (45deg) Delta Sigma value for each bin. In units of Msun/pc**2.
    cero_error : array, float 
        Bootstrap error in crossed Delta Sigma value for each bin. In units of Msun/pc**2.
        Computed only if nboot>0.
    stat_error : array, float
        Statistical error computed directly with the number of galaxies. Assumes an
        intrinsic ellipticity dispersion of 0.25. This error is always computed and should be
        used for testing only.
    N : array, int
        Number of galaxies stacked in each bin.
    sigma_critic : float
        Mean Sigma Critic of the stacked system. In units of Msun/pc**2.

    '''

    def __init__(self, data_L, data_S, rin=0.1, rout=10., nbins=10, scale=None, space='log',
        nboot=0, cosmo=cosmo, back_dz=0., precomputed_distances=True, njobs=1,
        colnames=None, lensing_coord='dang', covariance=False):
        
        #if not isinstance(cat, (CompressedCatalog, ExpandedCatalog)):
        #   raise TypeError('cat must be a LensCat catalog.')

        super().__init__(rin, rout, nbins, space, nboot, cosmo, back_dz)

        self.scale = scale
        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.lensing_coord = lensing_coord.lower()
        self.nlens = len(data_L)

        if colnames is None: colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}
        self.colnames = colnames

        # Compute the profile
        if data_S.index.name is not 'CATID':
            data_S_indexed = data_S.set_index('CATID')
        pf = self._profile(data_L=data_L, data_S=data_S_indexed)
        pf_final = self._reduce(pf)
        if covariance:
            cov = self._cov_matrix(pf)

        # Set the attributes
        self.sigma_critic = pf_final['sigma_critic']
        self.shear = pf_final['shear']
        self.cero = pf_final['cero']
        self.shear_error = pf_final['shear_error']
        self.cero_error = pf_final['cero_error']
        self.stat_error = pf_final['stat_error']
        self.r = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.N = pf_final['N'].astype(np.int32)
        self.cov = cov

    def _profile(self, data_L, data_S):
        ''' Computes profile for CompressedCatalog
        '''
        dict_per_lens = {'data_L': data_L, 'data_S': data_S, 'scale': self.scale,
                        'bins': self.bins, 'back_dz': self.back_dz, 'colnames': self.colnames,
                        'nboot': self.nboot, 'precomputed_distances': self.precomputed_distances,
                        'pf_flag': 'deltasigma', 'cosmo': self.cosmo, 'lcoord': self.lensing_coord}

        # Compute profiles per lens
        with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
            delayed_fun = delayed(_profile_per_lens)
            pf = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
        return pf

    def _reduce(self, pf):

        shear = np.sum( [jpf['shear_j'] for jpf in pf], axis=0)
        cero = np.sum( [jpf['cero_j'] for jpf in pf], axis=0)
        accum_w = np.sum( [jpf['accum_w_j'] for jpf in pf], axis=0)
        sigma_critic = np.sum( [jpf['sigma_critic_j'] for jpf in pf], axis=0)
        m_cal_num = np.sum( [jpf['m_cal_num_j'] for jpf in pf], axis=0)
        stat_error_num = np.sum( [jpf['stat_error_num_j'] for jpf in pf], axis=0)
        N = np.sum( [jpf['N_j'] for jpf in pf], axis=0)

        m_cal = m_cal_num / accum_w
        shear = (shear/accum_w) / m_cal
        cero = (cero/accum_w) / m_cal
        stat_error = np.sqrt(stat_error_num/accum_w**2) / m_cal
        sigma_critic = sigma_critic/accum_w

        x = {}
        x['shear'] = shear[0, :]
        x['cero'] = cero[0, :]
        x['sigma_critic'] = sigma_critic[0, :]
        x['stat_error'] = stat_error[0, :]
        x['shear_error'] = np.std(shear[1:,:], axis=0)
        x['cero_error'] = np.std(cero[1:,:], axis=0)
        x['N'] = N  
        return x

    def _cov_matrix(self, pf):
        '''Covariance matrix. Following Fang 2019 eq 5.'''
        
        m_arr = np.zeros((self.nlens, self.nbins))
        w_arr = np.zeros((self.nlens, self.nbins))
        shear_arr = np.zeros((self.nlens, self.nbins))

        for j, jpf in enumerate(pf):
            # Sum over all lenses
            m_arr += jpf['m_cal_num_j']
            w_arr += jpf['accum_w_j']
            shear_arr += jpf['shear_j']
            # Remove the j lens in the j row
            m_arr[j, :] -= jpf['m_cal_num_j']
            w_arr[j, :] -= jpf['accum_w_j']
            shear_arr[j, :] -= jpf['shear_j']      
        # Compute shear for each j realization without the j lens
        m_cal = m_arr / w_arr
        shear = (shear_arr/w_arr) / m_cal
        diff_shear = shear - shear.mean(axis=0)

        # Covariance
        cov = np.zeros((self.nbins, self.nbins))
        for k_ds in diff_shear:
            cov += k_ds.reshape(-1, 1) * k_ds
        cov *= (self.nlens - 1) / self.nlens
        return cov


@gentools.timer
class ShearProfile(Profile):
    '''Profile Constructor: DeltaSigma Profile
    Builds the radial shear profile using the lens and source catalog
    constructed with LensCat. Only works with the *compressed* catalog.

    Parameters
    ----------
    data_L : pandas dataframe
        Lens data attribute of LensCat.CompressedCatalog catalog, i.e: cat.data_L 
    data_S : pandas dataframe
        Source data attribute of LensCat.CompressedCatalog catalog, i.e: cat.data_S
    rin : float
        Inner radius to compute the binning. Units of Mpc. If scale parameter is
        specified the value should be in units (a factor) of scale. Default: 0.1.
    rout : float
        Outter radius to compute the binning. Units of Mpc. If scale parameter is
        specified the value should be in units (a factor) of scale. Default: 10.
    nbins : int
        Number of bins to compute the profile. Default: 10.
    scale : str
        Name of the column in data_L that will be used to scale the radial distances
        to the source galaxies. Should be in units of Mpc. Default: None.
    space : str
        String indicating if the radial binning should be logarithmic or linear.
        Possible values are: 'log' and 'lin'. Default: 'log'.
    nboot : int
        Number of bootstrap realizations to compute the error bars. Default: 0.
    cosmo : object
        Object indicating the cosmology. Must be instance an of astropy.cosmology.LambdaCDM
        Default: LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).
    back_dz : float
        Redshift gap between the lens and its sources. This will keep those galaxies
        with Zsources > Zlens + back_dz. Default: 0.
    precomputed_distances : bool
        Flag indicating if the lensing distances should be computed exactly or
        interpolated from a precomputed file.
    njobs : int
        Number of cores for parallelization. Default: 1.
    colnames : dict
        The lens column names for RA, DEC and Z in data_L are expected to be RA, DEC and Z.
        If that is not the case you can pass a dict with the new names.
        Example: colnames = {'RA': 'differentRAname', 'DEC': 'differentDECname', 'Z': 'Z'}.
        Default: None. This means colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}.
    lensing_coord : str
        Lensing coordinates, can be either 'DANG' or 'COMOV'.

    Attributes
    ----------
    r : array, float 
        Center of radial bins. If not scaled, in units of Mpc.
    shear : array, float
        Tangential Delta Sigma value for each bin. In units of shear (adimensional).
    shear_error : array, float
        Bootstrap error in tangential Delta Sigma value for each bin. In units of shear (adimensional).
        Computed only if nboot>0.
    cero : array, float
        Corssed (45deg) Delta Sigma value for each bin. In units of shear (adimensional).
    cero_error : array, float 
        Bootstrap error in crossed Delta Sigma value for each bin. In units of shear (adimensional).
        Computed only if nboot>0.
    stat_error : array, float
        Statistical error computed directly with the number of galaxies. Assumes an
        intrinsic ellipticity dispersion of 0.25. This error is always computed and should be
        used for testing only.
    N : array, int
        Number of galaxies stacked in each bin.
    sigma_critic : float
        Mean Sigma Critic of the stacked system. In units of Msun/pc**2.

    '''

    def __init__(self, data_L, data_S, rin=0.1, rout=10., nbins=10, space='log',
        nboot=0, cosmo=cosmo, back_dz=0., precomputed_distances=True, njobs=1, colnames=None, lensing_coord='dang'):
        
        #if not isinstance(cat, (CompressedCatalog, ExpandedCatalog)):
        #   raise TypeError('cat must be a LensCat catalog.')

        super().__init__(rin, rout, nbins, space, nboot, cosmo, back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances
        self.lensing_coord = lensing_coord.lower()

        if colnames is None: colnames = {'RA': 'RA', 'DEC': 'DEC', 'Z': 'Z'}
        self.colnames = colnames

        if data_S.index.name is not 'CATID':
            data_S_indexed = data_S.set_index('CATID')
        pf = self._profile(data_L=data_L, data_S=data_S_indexed)
        pf = self._reduce(pf)

        self.shear = pf['shear']
        self.cero = pf['cero']
        self.shear_error = pf['shear_error']
        self.cero_error = pf['cero_error']
        self.stat_error = pf['stat_error']
        self.r = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.N = pf['N'].astype(np.int32)
        self.sigma_critic = pf['sigma_critic']
        self.nlens = len(data_L)

    def _profile(self, data_L, data_S):
        ''' Computes profile for CompressedCatalog
        '''
        dict_per_lens = {'data_L': data_L, 'data_S': data_S, 'scale': self.scale,
                        'bins': self.bins, 'back_dz': self.back_dz, 'colnames': self.colnames,
                        'nboot': self.nboot, 'precomputed_distances': self.precomputed_distances,
                        'pf_flag': 'shear', 'cosmo': self.cosmo, 'lcoord': self.lensing_coord}

        # Compute profiles per lens
        with Parallel(n_jobs=self.njobs, require='sharedmem') as parallel:
            delayed_fun = delayed(_profile_per_lens)
            pf = parallel(delayed_fun(j, dict_per_lens) for j in range(len(data_L)))
        return pf

    def _reduce(self, pf):

        shear = np.sum( [pf[_]['shear_j'] for _ in range(len(pf))], axis=0)
        cero = np.sum( [pf[_]['cero_j'] for _ in range(len(pf))], axis=0)
        accum_w = np.sum( [pf[_]['accum_w_j'] for _ in range(len(pf))], axis=0)
        sigma_critic = np.sum( [pf[_]['sigma_critic_j'] for _ in range(len(pf))], axis=0)
        m_cal_num = np.sum( [pf[_]['m_cal_num_j'] for _ in range(len(pf))], axis=0)
        stat_error_num = np.sum( [pf[_]['stat_error_num_j'] for _ in range(len(pf))], axis=0)
        N = np.sum( [pf[_]['N_j'] for _ in range(len(pf))], axis=0)

        m_cal = m_cal_num / accum_w
        shear = (shear/accum_w) / m_cal
        cero = (cero/accum_w) / m_cal
        stat_error = np.sqrt(stat_error_num/accum_w**2) / m_cal
        sigma_critic = sigma_critic/accum_w

        x = {}
        x['shear'] = shear[0, :]
        x['cero'] = cero[0, :]
        x['sigma_critic'] = sigma_critic[0, :]
        x['stat_error'] = stat_error[0, :]
        x['shear_error'] = np.std(shear[1:,:], axis=0)
        x['cero_error'] = np.std(cero[1:,:], axis=0)
        x['N'] = N  
        return x



class ClampittProfile(Profile):

    def __init__(self, shear_map, rin=0.1, rout=10., nbins=10, space='log'):
        
        if not isinstance(cat, (Map, CompressedMap)):
            raise TypeError('cat must be a Shear Map instance.')

        super().__init__(rin, rout, nbins, space)
        
        region_map = self._get_region(shear_map)
        pf = self._profile(region_map)

        self.shear = pf['shear']
        self.cero = pf['cero']
        self.shear_error = pf['shear_error']
        self.cero_error = pf['cero_error']
        self.stat_error = pf['stat_error']
        self.r = pf['r']
        self.N = pf['N'].astype(np.int32)

    def _get_region(self, smp):

        box_size = smp.px.max() - smp.px.min()
        bin_sep = px.shape[0] / box_size

        xstart = px.shape[0]/2 - bin_sep/2
        xend = px.shape[0]/2 + bin_sep/2
        ystart = py.shape[0]/2 - bin_sep
        yend = py.shape[0]/2 + bin_sep

        region = np.zeros(px.shape, dtype=bool)
        region[ystart:yend, xstart:xend] = True
        rshape = (yend-ystart, xend-xstart)

        mp = Map()
        dx = smp.px[0,1]-smp.px[0,0]
        mp.bin_size = (dx, dx)
        for attr in ['shear', 'beta', 'shearx', 'sheary', 'shear1', 'shear2', 'stat_error', 'px', 'py']:
            mp[attr] = smp[attr][region].reshape(rshape)
        return mp

    def _profile(self, smp):

        nbins = int(smp.px.shape[1]/2)
        shear_fil = np.zeros(nbins)

        for j in range(nbins):  # coord y
            for i in range(nbins):   # coord x
                y = j - nbins
                x = i 
                shear_fil[j] = smp.shear[j, i] + smp.shear[y, -x]




##########################################################################################
# Deprecated

@gentools.timer
class ExpandedProfile(Profile):

    def __init__(self, data=None, rin_hMpc=0.1, rout_hMpc=10., nbins=10, space='log', 
        nboot=0, cosmo=cosmo, back_dz=0., precomputed_distances=True, njobs=1):
        
        super().__init__(rin_hMpc, rout_hMpc, nbins, space, nboot, cosmo, back_dz)

        self.njobs = njobs
        self.precomputed_distances = precomputed_distances

        pf = self._profile(data=data)
        pf = self._reduce(pf)
        
        # Now in units of h*Msun/pc**2
        self.shear = pf['shear']/self.cosmo.h
        self.cero = pf['cero']/self.cosmo.h
        self.shear_error = pf['shear_error']/self.cosmo.h
        self.cero_error = pf['cero_error']/self.cosmo.h
        self.stat_error = pf['stat_error']/self.cosmo.h
        self.r_hMpc = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.N = pf['N'].astype(np.int32)


    def _profile(self, data):
        ''' Computes profile for ExpandedCatalog
        '''
        #if self.back_dz != 0.:
        mask_dz = data['Z_B'].values >= data['Z'].values + self.back_dz
        data = data[mask_dz]

        DD = gentools.compute_lensing_distances(zl=data['Z'].values, zs=data['Z_B'].values,
            precomputed=self.precomputed_distances, cosmo=self.cosmo)
        
        Mpc_scale = gentools.Mpc_scale(dl=DD['DL'])
        sigma_critic = gentools.sigma_critic(dl=DD['DL'], ds=DD['DS'], dls=DD['DLS'])

        # Compute distance and ellipticity components...
        dist, theta = gentools.sphere_angular_vector(data['RAJ2000'].values, data['DECJ2000'].values,
                                                    data['RA'].values, data['DEC'].values, units='deg')
        theta += 90. 
        dist_hMpc = dist*3600. * Mpc_scale*cosmo.h # radial distance to the lens centre in Mpc/h
        et, ex = gentools.polar_rotation(data['e1'].values, data['e2'].values, np.deg2rad(theta))

        digit = np.digitize(dist_hMpc, bins=self.bins)-1
            
        dict_per_bin = {'m': data['m'].values,
                        'weight': data['weight'].values,
                        'digit': digit, 
                        'sigma_critic': sigma_critic.values,
                        'et': et.values, 'ex': ex.values,
                        'nboot': self.nboot}

        with Parallel(n_jobs=self.njobs) as parallel:
            delayed_fun = delayed(_profile_per_bin)
            pf = parallel(delayed_fun(i, dict_per_bin) for i in range(self.nbins))
        return pf

    def _reduce(self, pf):

        shear = np.array( [pf[_]['shear_i'] for _ in range(len(pf))] ).T
        cero = np.array( [pf[_]['cero_i'] for _ in range(len(pf))] ).T
        accum_w = np.array( [pf[_]['accum_w_i'] for _ in range(len(pf))] ).T
        m_cal_num = np.array( [pf[_]['m_cal_num_i'] for _ in range(len(pf))] ).T
        stat_error_num = np.array( [pf[_]['stat_error_num_i'] for _ in range(len(pf))] ).T
        N = np.array( [pf[_]['N_i'] for _ in range(len(pf))] )

        m_cal = m_cal_num / accum_w
        shear = (shear/accum_w) / m_cal
        cero = (cero/accum_w) / m_cal
        stat_error = np.sqrt(stat_error_num/accum_w**2) / m_cal

        x = {}
        x['shear'] = shear[0, :]
        x['cero'] = cero[0, :]
        x['stat_error'] = stat_error[0, :]
        x['shear_error'] = np.std(shear[1:,:], axis=0)
        x['cero_error'] = np.std(cero[1:,:], axis=0)
        x['N'] = N.astype(np.int32)
        return x

