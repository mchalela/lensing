import numpy as np
#from pylab import *

from astropy import units
from astropy.coordinates import SkyCoord
from scipy import spatial
#import circles
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sdss = np.load('SDSS/grispy_sdssDR15.npy')


def box_region(m, lon_min, lon_max, lat_min, lat_max, **kwargs):
	N=100
	lons = np.linspace(lon_min,lon_max, N)
	lats = np.linspace(lat_min,lat_max, N)
	ones = np.ones(N)

	lons = np.concatenate((ones*lon_min, lons,
							ones*lon_max, lons[::-1]))
	lats = np.concatenate((lats, ones*lat_max,
							lats[::-1], ones*lat_min))

	x, y = m( lons, lats )
	xy = zip(x,y)
	#return xy
	poly = Polygon( xy, **kwargs)#facecolor='red', alpha=0.4 )
	plt.gca().add_patch(poly)

#sdss_fields = np.loadtxt('sdss_rerun301.txt')

#~ print len(M2)+len(M3)+len(S), len(sdss)
AR= sdss[::100,0]
DEC= sdss[::100,1]
#AR= sdss_fields[::10,0]
#DEC= sdss_fields[::10,1]


#cs82_AR = np.random.rand(10000)*80 - 40.
#cs82_DEC = np.random.rand(10000)*3 - 1.5

# KiDS region
#kidsN = np.concatenate((region(156., 225., -5., 4.),
#						region(225., 238., -3., 4.),
#						region(128., 142., -2., 3.)))
#kidsN_AR = np.random.rand(10000)*82 + 156. #156° ― 225°
#kidsN_DEC = np.random.rand(10000)*9 -5. 	#−5° ― +4°

# Make plot
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['legend.fontsize'] = 12.0

fig, ax = plt.subplots(1, 1)
m = Basemap(projection='moll',lon_0=270,resolution='c')
meridians=np.arange(0.,390.,30.)
parallels=np.arange(-90.,120.,30.)
m.drawparallels(parallels,labels=[1,0,0,0], fontsize=12)
m.drawmeridians(meridians)



# Galactic plane
xy_e=np.loadtxt('Extinction/E_BV_grid_0100_0500.dat').flatten()
xy = np.loadtxt('Extinction/l_b_grid_0100_0500.dat')
x_l=xy[:,::2].flatten()
y_b=xy[:,1::2].flatten()
c = SkyCoord(l=x_l*units.degree, b=y_b*units.degree, frame='galactic')
gal_ra,gal_dec = c.icrs.ra, c.icrs.dec
x_gal,y_gal = m(gal_ra,gal_dec)
i_gal = np.argsort(x_gal)
x_gal,y_gal = x_gal[i_gal],y_gal[i_gal]
xy_e = xy_e[i_gal]
xy_e[xy_e>3.5] = 3.5
#xy_e[(xy_e>2)*(xy_e<5)] = 2
xy_e[(xy_e<1)] = 0
m.scatter(x_gal,y_gal,marker='.',c=xy_e, s=5,lw=1, alpha=0.4,cmap=plt.cm.afmhot_r)
plt.clim(0,4)
#m.plot(x_gal,y_gal,'.',c=xy_e, lw=1, alpha=0.01, label='MW')

'''
gal_lon = np.linspace(0,360,500)
gal_lat = np.zeros(500)
c = SkyCoord(l=gal_lon*units.degree, b=gal_lat*units.degree, frame='galactic')
gal_ra,gal_dec = c.icrs.ra, c.icrs.dec
x_gal,y_gal = m(gal_ra,gal_dec)
i_gal = np.argsort(x_gal)
x_gal,y_gal = x_gal[i_gal],y_gal[i_gal]
m.plot(x_gal,y_gal,'-',c='firebrick', lw=1, alpha=1, label='MW')
'''

# SDSS points
x_sdss,y_sdss = m(AR,DEC)
m.scatter(x_sdss,y_sdss,marker='s',c='silver', s=5,lw=1, alpha=0.2, label='SDSS-DR15')

# Stripe82 region
c_s = 'seagreen'#'indianred'
box_region(m, -40., 40., -2, 2, facecolor=c_s, edgecolor=c_s, lw=2, alpha=1, label='Stripe82')
#plt.fill_between(x_cs82[:-1], y_low_cs82[:-1], y_up_cs82[:-1], color='indianred')
'''
# KiDS region
c_k = 'royalblue'#'mediumseagreen'
box_region(m, 156., 225., -5., 4., facecolor=c_k, edgecolor=c_k, lw=2,alpha=1, label='KiDS')
box_region(m, 225., 238., -3., 4., facecolor=c_k, edgecolor=c_k, lw=2, alpha=1)
box_region(m, 128., 142., -2., 3., facecolor=c_k, edgecolor=c_k, lw=2, alpha=1)

# CFHT region
c_c = 'm'#'mediumorchid'
box_region(m, 30., 39., -11.5, -3.5, facecolor=c_c, edgecolor=c_c, alpha=0.7, lw=2, label='CFHT-W1-4')
box_region(m, 131., 137., -7., -1., facecolor=c_c, edgecolor=c_c, alpha=0.7, lw=2)
box_region(m, 208., 221., 51., 58., facecolor=c_c, edgecolor=c_c, alpha=0.7, lw=2)
box_region(m, 330., 336., -1., 5., facecolor=c_c, edgecolor=c_c, alpha=0.7, lw=2)
'''




plt.xlabel('Equatorial ($\\alpha$, $\\delta$)', fontsize=12)
plt.legend(loc=4, fontsize=10,framealpha=1)
plt.tight_layout()
