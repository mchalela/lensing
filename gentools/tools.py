import time
import numpy as np
from functools import wraps
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

cvel = 299792458. 	# Speed of light (m.s-1)
G    = 6.670e-11   	# Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16 	# 1 pc (m)
Msun = 1.989e30 	# Solar mass (kg)

def Mpc_scale(dl):
	MpcScale = dl*np.deg2rad(1./3600.)
	return MpcScale

def Mpc2deg(R_Mpc, z, cosmo=cosmo):
	dl = np.array(cosmo.angular_diameter_distance(z))
	MpcScale = Mpc_scale(dl)
	R_deg = (R_Mpc/MpcScale)/3600.
	return R_deg
	
def deg2Mpc(R_deg, z, cosmo=cosmo):
	dl = np.array(cosmo.angular_diameter_distance(z))
	MpcScale = Mpc_scale(dl)
	R_Mpc = R_deg*3600.*MpcScale
	return R_Mpc

def sigma_critic(dl, ds, dls):
	beta = dls/ds
	SigmaCritic = cvel**2/(4.*np.pi*G*(dl*1e6*pc)) * (1./beta) * (pc**2/Msun)
	#beta_mean = beta.mean()
	#sigma_critic_mean = SigmaCritic.mean()
	return SigmaCritic

def make_bins(rin, rout, nbins, space):	
	if space=='log':
		bins = np.geomspace(rin, rout, nbins+1)
	else:
		bins = np.linspace(rin, rout, nbins+1)
	return bins

def digitize(data, bins):
    """Return data bin index."""
    N = len(bins) - 1
    digit = (N * (data - bins[0]) / (bins[-1] - bins[0])).astype(np.int)
    return digit

def skycoord_average(ra, dec):
    """Compute the mean position of points in a sphere."""
    from astropy import units as u 
    from astropy.coordinates import SkyCoord

    pos = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    car = pos.cartesian.mean()
    new = SkyCoord(car)
    ra_mean = new.ra.value
    dec_mean = new.dec.value
    return ra_mean, dec_mean


# General tools for the lensing module
class classonly(classmethod):
    def __get__(self, obj, type):
        if obj: raise AttributeError
        return super(classonly, self).__get__(obj, type)

def seconds2str(dt):
    '''Convert seconds to a printable format'''
    h, m, s = int(dt//3600), int((dt%3600) // 60), dt % 60
    if dt > 60.:
    	_str = f'{s:.1f}s'
    elif dt > 1.:
    	_str = f'{s:.2f}s'
    elif dt > 1e-3:
        ms = int(s*1e3)	# miliseconds
        _str = f'{ms:d}ms'
    elif dt > 1e-6:
    	mus = int(s*1e6)	# microseconds
    	_str = f'{mus:d}\u03BCs'

    if h != 0: 
        _str = f'{h:d}h{m:02d}m' + _str
    elif m !=0:
        _str = f'{m:d}m' + _str
    return _str	

def timer(method):
    '''Decorator to time a function runtime'''
    @wraps(method)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        output = method(*args, **kwargs)
        dt = time.time() - t0
        dt = seconds2str(dt)
        print('<{} finished in {}>'.format(method.__name__, dt))
        return output
    return wrapper