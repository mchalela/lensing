import os
import numpy as np
import scipy.sparse
from astropy.cosmology import FLRW

def sphere_angular_separation(lon1, lat1, lon2, lat2):
    '''
    Angular separation between two points on a sphere

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : Angle, Quantity or float
        Longitude and latitude of the two points.  Quantities should be in
        angular units; floats in radians

    Returns
    -------
    angular separation : Quantity or float
        Type depends on input; Quantity in angular units, or float in radians

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1],
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    [1] http://en.wikipedia.org/wiki/Great-circle_distance
    '''
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    sep = np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator)
    return sep



def sphere_angular_vector(ra, dec, ra_center, dec_center, units='rad'): 
    '''
    Angular separation and orientation between two points on a sphere.
    Orientation is meassured from declination axis clockwise.

    Parameters
    ----------
    ra, dec, ra_center, dec_center : Angle, Quantity or float
        Right ascension and declination of the two points.
    units : String indicating ra and dec units.
        Options are: 'rad', 'deg'. Default: 'rad'.

    Returns
    -------
    distance, orientation : Quantity or float
        Polar vector on the sphere in units defined by 'units'.
    '''
    if units not in ['rad', 'deg']:
        raise ValueError('Argument units="{}" not recognized. '
            'Options are: "rad", "deg".'.format(units))
        
    if units == 'deg':
        ra, dec = np.deg2rad(ra), np.deg2rad(dec)
        ra_center, dec_center = np.deg2rad(ra_center), np.deg2rad(dec_center)

    #center all points in ra
    ra_prime = ra-ra_center
    ra_center = 0
    #convert to positive values of RA
    negative = (ra_prime<0)
    ra_prime[negative] += 2*np.pi
    #define wich quadrant is each point
    Q1 = (ra_prime<np.pi) & (dec-dec_center>0)
    Q2 = (ra_prime<np.pi) & (dec-dec_center<0)
    Q3 = (ra_prime>np.pi) & (dec-dec_center<0)
    Q4 = (ra_prime>np.pi) & (dec-dec_center>0)
    
    #Calculate the distance between the center and object, and the azimuthal angle
    dist = sphere_angular_separation(ra_prime, dec, ra_center, dec_center)
        
    #build a triangle to calculate the spherical cosine law
    x = sphere_angular_separation(ra_center, dec, ra_center, dec_center)
    y = sphere_angular_separation(ra_prime, dec, ra_center, dec)
    
    #Apply shperical cosine law
    cos_theta = (np.cos(y) - np.cos(x)*np.cos(dist))/(np.sin(x)*np.sin(dist))
    #Round cosines that went over because of rounding errors
    round_high = (cos_theta >  1)
    round_low  = (cos_theta < -1)
    cos_theta[round_high] =  1 
    cos_theta[round_low]  = -1
    theta = np.arccos(cos_theta)
    #Correct the angle for quadrant (because the above law calculates the acute angle between the "horizontal-RA" direction
    #and the angular separation great circle between the center and the object

    #theta[Q1] = theta[Q1]    # First quadrant remains the same
    theta[Q2] = np.pi - theta[Q2] 
    theta[Q3] = np.pi + theta[Q3]
    theta[Q4] = 2*np.pi - theta[Q4]

    if units == 'deg':
       dist, theta = np.rad2deg(dist), np.rad2deg(theta)

    return dist, theta


def _precompute_lensing_distances(zl_max, zs_max, dz=0.0005, cosmo=None):
    '''Precompute lensing distances DL, DS, DLS and save to a file

    Parameters
    ----------
    zl_max, zs_max : float
        Maximum redshift to wich the distances will be computed
    dz : float
        Redshift step
    cosmo: cosmology
        instance of astropy.cosmology.FLRW, like Planck15 or LambdaCDM. 

    Returns
    -------
    file : string
        Filename where the sparse matrix was saved. The format is
        'PrecomputedDistances_dz_{}.npz'.format(dz)

    '''
    if not isinstance(cosmo, FLRW):
        raise TypeError, 'cosmo is not an instance of astropy.cosmology.FLRW' 
    zl = np.arange(0., zl_max+dz, dz)
    zs = np.arange(0., zs_max+dz, dz)

    B = scipy.sparse.lil_matrix((len(zl), len(zs)), dtype=np.float64)
    for i in xrange(len(zl)):
            B[i, i:] = cosmo.angular_diameter_distance_z1z2(zl[i], zs[i:]).value

    path = os.path.dirname(os.path.abspath(__file__))+'/'
    filename = 'PrecomputedDistances_dz_{}.npz'.format(dz)
    scipy.sparse.save_npz(path+filename, scipy.sparse.csc_matrix(B))
    return path+filename


#-----------------------------------------------------------------------
# Recuperamos los datos
def compute_lensing_distances(zl, zs, precomputed=False, dz=0.0005, cosmo=None):
    '''
    Compute lensing Angular diameter distances.

    Parameters
    ----------
    zl, zs : array, float
        Redshift of the lens and the source.
    precomputed : bool
        If False the true distances are computed. If False the distances
        will be interpolated from a precomputed file.
    dz : float
        step of the precomputed distances file. If precomputed is True,
        this value will be used to open the file:
            'PrecomputedDistances_dz_{}.npz'
    cosmo: cosmology
        instance of astropy.cosmology.FLRW, like Planck15 or LambdaCDM.

    Returns
    -------
    DL, DS, DLS : array, float
        Angular diameter distances. DL: dist. to the lens, DS: dist. to
        the source, DLS: dist. from lens to source. 
    '''
    if not precomputed:
        if not isinstance(cosmo, FLRW):
            raise TypeError, 'cosmo is not an instance of astropy.cosmology.FLRW'           
        DL  = cosmo.angular_diameter_distance(zl).value
        DS  = cosmo.angular_diameter_distance(zs).value
        DLS = cosmo.angular_diameter_distance_z1z2(zl, zs).value

    else:
        path = os.path.dirname(os.path.abspath(__file__))+'/'
        H = scipy.sparse.load_npz(path+'PrecomputedDistances_dz_{}.npz'.format(dz)).todense()
        H = np.asarray(H)
        Delta_z = dz
        zl_big = zl/Delta_z
        zs_big = zs/Delta_z
        zl_idx = zl_big.astype(np.int32)
        zs_idx = zs_big.astype(np.int32)

        zl1_frac = (zl_big - zl_idx)*Delta_z
        zs1_frac = (zs_big - zs_idx)*Delta_z
        zl2_frac = (zl_idx+1 - zl_big)*Delta_z
        zs2_frac = (zs_idx+1 - zs_big)*Delta_z
        # Lineal interpolation for DL and DS
        DL  = (H[0, zl_idx]*zl2_frac + H[0, zl_idx+1]*zl1_frac) / Delta_z
        DS  = (H[0, zs_idx]*zs2_frac + H[0, zs_idx+1]*zs1_frac) / Delta_z
        # Bilineal interpolation for DLS
        A = H[zl_idx, zs_idx]*zl2_frac*zs2_frac
        B = H[zl_idx+1, zs_idx]*zl1_frac*zs2_frac
        C = H[zl_idx, zs_idx+1]*zl2_frac*zs1_frac
        D = H[zl_idx+1, zs_idx+1]*zl1_frac*zs1_frac
        DLS = (A + B + C + D) / Delta_z**2

    return [DL, DS, DLS]