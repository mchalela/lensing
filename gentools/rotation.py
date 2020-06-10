import numpy as np 

def polar_rotation(e1, e2, theta):
    '''
    Rotate ellipticity components
    '''
    e1_rot = e1*np.cos(2*theta)-e2*np.sin(2*theta)
    e2_rot = e1*np.sin(2*theta)+e2*np.cos(2*theta)
    return e1_rot, e2_rot


def equatorial_coordinates_rotation(ra, dec, ra_center, dec_center, angle, units='deg'):
    '''
    Rotation in spherical coordinates using Rodrigues formula.

    Parameters
    ----------
    ra, dec, ra_center, dec_center : Array or float
        Right ascension and declination of the points to rotate
        and their pivot centers.
    units : String indicating ra, dec and angle units.
        Options are: 'rad', 'deg'. Default: 'deg'.

    Returns
    -------
    ra_rot, dec_rot : Array or float
        Right ascencion and declination after the desired rotation
        in units defined by 'units'.

    Notes
    -----
    See this equations: https://math.stackexchange.com/a/1404353
    In the prevoius link, the angle definition is not explicit
    but they are like this:
     > phi is longitude, or right ascension.
     > theta is the zenith angle, or 90 - declination
    '''
    if units not in ['rad', 'deg']:
        raise ValueError('Argument units="{}" not recognized. '
            'Options are: "rad", "deg".'.format(units))

    if units == 'deg':
        ra, dec = np.deg2rad(ra), np.deg2rad(dec)
        ra_center, dec_center = np.deg2rad(ra_center), np.deg2rad(dec_center)
        angle = np.deg2rad(angle)

    phi = ra
    theta = dec
    theta = np.pi/2. - theta                # theta is measured from the zenith

    ax_phi = ra_center
    ax_theta = dec_center
    ax_theta = np.pi/2. - ax_theta          # ax_theta is measured from the zenith

    ax = np.sin(theta)*np.cos(phi)
    ay = np.sin(theta)*np.sin(phi)
    az = np.cos(theta)
    kx = np.sin(ax_theta)*np.cos(ax_phi)
    ky = np.sin(ax_theta)*np.sin(ax_phi)
    kz = np.cos(ax_theta)

    a = np.stack((ax, ay, az), axis=1)
    k = np.stack((kx, ky, kz), axis=1)

    dot = np.sum(k*a, axis=1).reshape((-1, 1))
    cross = np.cross(k, a, axisa=1, axisb=1)

    b = a*np.cos(angle) + cross*np.sin(angle) + k*dot*(1-np.cos(angle))

    new_phi = np.arctan2(b[:,1], b[:,0])
    new_theta = np.arctan2( np.sqrt(b[:,0]**2 + b[:,1]**2), b[:,2] )
    new_theta = np.pi/2 - new_theta          # get back to measure theta from equator

    if units == 'deg':
        ra_rot, dec_rot = np.rad2deg(new_phi), np.rad2deg(new_theta)
    else:
        ra_rot, dec_rot = new_phi, new_theta

    return ra_rot, dec_rot