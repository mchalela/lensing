#import os, platform
import numpy as np
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def Mpc2deg(R_Mpc, z, cosmo=cosmo):
	dl = np.array(cosmo.angular_diameter_distance(z))
	Mpcscale=dl*np.deg2rad(1.0/3600.0)
	R_deg = (R_Mpc/Mpcscale)/3600.
	return R_deg
	
def deg2Mpc(R_deg, z, cosmo=cosmo):
	dl = np.array(cosmo.angular_diameter_distance(z))
	Mpcscale=dl*np.deg2rad(1.0/3600.0)
	R_Mpc = R_deg*3600.*Mpcscale
	return R_Mpc