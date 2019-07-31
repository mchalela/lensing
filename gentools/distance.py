import numpy as np

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
    Angular separation and orientation between two points on a sphere

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