import numpy as np 

def polar_rotation(e1, e2, theta):
	'''
	Rotate ellipticity components
	'''
	e1_rot = -e1*np.cos(2*theta)-e2*np.sin(2*theta)
	e2_rot = -e1*np.sin(2*theta)+e2*np.cos(2*theta)
	return e1_rot, e2_rot