import os
from astropy.modeling import Model, models#, fitting 
from astropy.cosmology import Planck15 #, LambdaCDM
from astropy import units
import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import CubicSpline
import scipy.special as sp
try:
	import camb
except Exception:
	print 'WARNING: [lensing.densmodel] No module named camb. Used in second halo profile.'


cvel= 299792458.		# Speed of light (m.s-1)
G 	= 6.67384e-11		# Gravitational constant (m3.kg-1.s-2)
pc 	= 3.085678e16		# 1 pc (m)
Msun= 1.989e30			# Solar mass (kg)
#cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.7}


# Density models ----------------------
class Density:
	'''
	Definition of contrast density models.
	'''

	def __init__(self, z, cosmo=None):

		self.z = z
		if cosmo is None:
			self.cosmo = Planck15
			print 'Assuming Planck15 cosmology: ', self.cosmo
		else:
			self.cosmo = cosmo

		self.J2 = None 		# Creates an empty variable for the Bessel function

	def Bar(self, r_h, logMstar_h=13.):
		'''
		Barionic density contrast profile. r_h and Mstar_h must be in Mpc/h and Msun/h respectively
		'''
		Mstar_h = 10**logMstar_h
		rp=r_h*1.e6
		bcg_density = Mstar_h/(np.pi*rp**2)			# Density contrast in h*Msun/pc2
		return bcg_density

	def Bar_with_M200(self, r_h, logM200_h=13.):
		'''
		Wraper to fit the Barionic mass with the M200 of the halo instead of Mstar.
		We use the scaling relation of Johnston et a. 2007. Section 5.4
		https://arxiv.org/pdf/0709.1159.pdf
		'''
		M200_h = 10**logM200_h
		p0 = 1.334e12 		# in Msun/h
		p1 = 6.717e13 		# in Msun/h
		p2 = -1.380
		Mstar_h = p0 / (1 + (M200_h/p1)**p2)
		
		bcg_density = self.Bar(r_h, logMstar_h=np.log10(Mstar_h))
		return bcg_density

	def SIS(self, r_h, disp=300.):
		'''
		SIS density contrast profile. r_h and disp must be in Mpc/h and km/s respectively
		'''
		rm=r_h*1.e6*pc
		sis_density = (disp*1.e3)**2 / (2.*G*rm) * pc**2/Msun		# Density contrast in h*Msun/pc2
		return sis_density

	def SISoff(self):
		return None

	def NFW(self, r_h, logM200_h=13.):
		'''
		NFW density contrast profile. r_h and M200_h must be in Mpc/h and Msun/h respectively
		'''
		M200_h = 10**logM200_h

		rhoc = self.cosmo.critical_density(self.z).si.value / self.cosmo.h**2	#critical density at z (h2.kg.m-3)
		rhoc_mpc = rhoc * (pc*1.0e6)**3.0
		
		R200 = np.cbrt(M200_h*3.0*Msun/(800.0*np.pi*rhoc_mpc))
		
		#calculo de c usando la relacion de Duffy et al 2008. https://arxiv.org/pdf/0804.2486.pdf
		c = 5.71 * (M200_h/2.e12)**-0.084 * (1.+self.z)**-0.47
		
		deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
		x=(r_h*c)/R200
		m1=x< 1.0
		atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
		jota=np.zeros(len(x))
		jota[m1]=(4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
			+ (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
			+ (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))
		m2=x> 1.0     
		atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
		jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
			+ (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
			+ (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
		m3=(x == 1.0)
		jota[m3]=2.0*np.log(0.5)+5.0/3.0
		rs_m=(R200*1.e6*pc)/c
		kapak=(2.*rs_m*deltac*rhoc)*(pc**2/Msun)

		nfw_density = kapak*jota		# Density contrast in h*Msun/pc2
		return nfw_density

	def NFWoff(self, r_h, logM200_h=13., disp_offset_h=0.1):
		'''
		NFW density contrast profile with a gaussian miscentered kernel.
		r_h and disp_offset_h must be in Mpc/h
		M200_h must be in Msun/h
		'''
		M200_h = 10**logM200_h

		rhoc = self.cosmo.critical_density(self.z).si.value / self.cosmo.h**2	#critical density at z (h2.kg.m-3)
		rhoc_mpc = rhoc * (pc*1.0e6)**3.0
		
		R200 = np.cbrt(M200_h*3.0*Msun/(800.0*np.pi*rhoc_mpc))
		
		#calculo de c usando la relacion de Duffy et al 2008. https://arxiv.org/pdf/0804.2486.pdf
		c = 5.71 * (M200_h/2.e12)**-0.084 * (1.+self.z)**-0.47
		
		rs = R200/c
		deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))	
		
		#epsabs=1.49e-04 ; epsrel=1.49e-06
		
		def _sigma_nfw(x): #Ecuacion 11 Wright and Brainerd 2000	

			sigma=np.zeros(len(x))
			m1 = x<1.0
			atanh = np.arctanh(((1.-x[m1])/(1.+x[m1]))**0.5)
			sigma[m1] = (1. - (2./(1.-x[m1]**2)**0.5)*atanh)/(x[m1]**2-1.)
			m2 = x>1.0 
			atan = np.arctan(((x[m2]-1.)/(1.+x[m2]))**0.5)
			sigma[m2] = (1. - (2./(x[m2]**2-1.)**0.5)*atan)/(x[m2]**2-1.)
			m3 = x == 1.0
			sigma[m3] = 1./3.

			return sigma #*kapak     #quito kapak aca, y la agrego en delta_sigma

		def _sigma_theta(theta, x, r_off): #Para calcular la integral en theta	
			x_off = ((x**2 + r_off**2 - 2*x*r_off*np.cos(theta))**0.5)/rs
			return _sigma_nfw(x_off)
							
		def _sigma_off(r_off, x):
			err_max=1e0
			err_min=1e-1
			epsrel=x/(10+x)*(err_max-err_min)+err_min
			th = np.linspace(0,2*np.pi,1000)
			int_sigma_theta = simps(_sigma_theta(th, x, r_off), th, even='last')/(2*np.pi)
			#int_sigma_theta = quad(lambda theta: _sigma_theta(theta, x, r_off), 0., 2*np.pi, epsabs=0, epsrel=epsrel)[0]/(2*np.pi)
			P = (r_off/(disp_offset_h**2)) * np.exp(((r_off/disp_offset_h)**2.)/(-2.))
			return P*int_sigma_theta
			
		def _sigma_bar_off(x):
			err_max=1e-3
			err_min=1e-3
			epsrel=x/(10+x)*(err_max-err_min)+err_min
			int_sigma_r	= quad(lambda r_off: _sigma_off(r_off, x), 0., np.inf, epsabs=0, epsrel=epsrel)[0]
			return int_sigma_r

		BIN=len(r_h)
		delta_sigma_off=[]
		for i in xrange(BIN):
			x=r_h[i]
			err_max=1e-1
			err_min=1e-2
			epsrel=x/(10+x)*(err_max-err_min)+err_min
			sigma_bar_off = quad(lambda x_i: (_sigma_bar_off(x_i)*x_i), 0., x, epsabs=0, epsrel=epsrel)[0] *2. / (x**2)
			sigma_off = quad(lambda r_off: _sigma_off(r_off, x), 0., np.inf, epsabs=0, epsrel=epsrel)[0]
			rs_m=rs*1.e6*pc
			kapak=(2.*rs_m*deltac*rhoc)*(pc**2/Msun)				
			delta_sigma_off.append((sigma_bar_off - sigma_off)*kapak)		# Density contrast in h*Msun/pc2
			
		return np.asarray(delta_sigma_off)

	def NFWCombined(self, r_h, logM200_h=13., disp_offset_h=0.1, p_cen=1.):
		'''
		Wraper to combine centered and miscentered NFW.
		r_h and disp_offset_h must be in Mpc/h
		M200_h must be in Msun/h		
		'''
		nfw_cen = self.NFW(r_h, logM200_h=logM200_h)
		nfw_off = self.NFWoff(r_h, logM200_h=logM200_h, disp_offset_h=disp_offset_h)
		nfw_off = nfw_off.reshape(nfw_cen.shape)
		nfw_combined = p_cen * nfw_cen + (1-p_cen) * nfw_off
		return nfw_combined

	def _set_secondhalo_params(self, r_h):
		'''
		Precomputes some parameters
		'''
		kh_npoints = 1e7 		# Something like 1e7 
		self.kh_min = 1e-4
		self.kh_max = 1e4
		self.kh_bins = np.geomspace(self.kh_min, self.kh_max, kh_npoints, endpoint=True)

		Dang_h = self.cosmo.angular_diameter_distance(self.z).value * self.cosmo.h 	# in Mpc/h
		self.l_bins = self.kh_bins * (1+self.z)*Dang_h

		theta = r_h/Dang_h
		self.J2 = sp.jn(2, self.l_bins*theta[:,np.newaxis])

		return None #J2


	def SecondHalo(self, r_h, logM200_h=13., Delta=200.):
		'''
		Computes the second halo term. r_h and M200_h must be in Mpc/h and Msun/h respectively
		'''
		M200_h = 10**logM200_h
		# These are for the integration
		if self.J2 == None:
			self._set_secondhalo_params(r_h)			
		#kh_min = 1e-4
		#kh_max = 1e4
		#kh_npoints = 1e7 		# Something like 1e7 
		#kh_bins = np.geomspace(kh_min, kh_max, kh_npoints, endpoint=True)

		def _power_spectrum_camb(z):
			'''
			Computes the matter power spectrum using CAMB (https://camb.info/)
			'''

			table_path = './'
			table_name = table_path + 'power_spectrum_z%.2f.dat' % round(z,2)

			# Create the table if it doesnt exist-----------------------------------------------------------------
			if not os.path.isfile(table_name):

				camb_cosmo = {'H0': self.cosmo.H0.value,
							  'ombh2': self.cosmo.Ob0 * self.cosmo.h**2,
							  'omch2': self.cosmo.Odm0 * self.cosmo.h**2,
							  'TCMB': self.cosmo.Tcmb0.value,
							  'omk': self.cosmo.Ok0,
							  'mnu': self.cosmo.m_nu.value.sum(),
							  'nnu': self.cosmo.Neff}

				#Now get matter power spectra at redshift z
				pars = camb.CAMBparams()
				pars.set_cosmology(**camb_cosmo)
				#pars.set_cosmology(H0=100.*h, ombh2=0.0486*h**2, omch2=0.2584*h**2)
				#pars.set_cosmology(H0=100., ombh2=0.022, omch2=0.122)
				pars.InitPower.set_params(ns=0.965)
				#Not non-linear corrections couples to smaller scales than you want
				pars.set_matter_power(redshifts=[z], kmax=self.kh_max)
				#pars.set_matter_power(redshifts=[0.2], kmax=kh_max)

				#Linear spectra
				pars.NonLinear = camb.model.NonLinear_none
				print 'P(k): get_results...'
				results = camb.get_results(pars)
				print 'P(k): get_matter_power_spectrum...'
				kh, zz, Pk = results.get_matter_power_spectrum(minkh=self.kh_min, maxkh=self.kh_max, npoints = 10**5)

				# Columns: kh (h/Mpc)	Pk (Mpc^3/h^3)
				table = np.vstack((kh, Pk[0,:])).T
				np.savetxt(table_name, table)
			else:
				table = np.loadtxt(table_name)

			kh = table[:,0]
			Pk = table[:,1]

			return kh, Pk

		# ------------------------------------------------------------------------------------------------------
		# In this section we compute the bias factor from Tinker et al 2010.
		# First we need the 'peak height' nu.

		def _bias_factor(M200_h, z, Delta):
			'''
			Computes the bias factor as parametrized by Tinker et al 2010 using the peak height nu.
			Given that it requires a heavy integral we pre-compute a table (Mass, nu)
			and interpolate with a cubic spline for a given virial mass M200 (Msun/h).
			'''

			table_path = './'
			table_name = table_path + 'matter_variance_z%.2f.dat' % round(z,2)

			# Create the table if it doesnt exist-----------------------------------------------------------------
			if not os.path.isfile(table_name):
				# Define some parameters
				rhoc = self.cosmo.critical_density(z)
				rhoc = rhoc.to(units.Msun/units.Mpc**3).value / self.cosmo.h**2			#critical density (h^2 Msun Mpc^-3)
				rhom = rhoc * self.cosmo.Om(z)		#mean density at z

				delta_c = 1.686		# From Tinker et al 2010
				R_npoints = 100
				R_min = 0.05		# in Mpc/h
				R_max = 50.			# in Mpc/h
				R_bins = np.geomspace(R_min, R_max, R_npoints, endpoint=True)

				kh_0, Pk_0 = _power_spectrum_camb(z)

				# Interpolates for the actual bins we want
				cs = CubicSpline(kh_0,Pk_0)
				Pk = cs(self.kh_bins)

				delta_k = self.kh_bins[1:] - self.kh_bins[:-1]
				k2_Pk = self.kh_bins*self.kh_bins*Pk
				k2_Pk_next = k2_Pk[1:]

				print 'Bias integral...'
				I = []
				for R_i in R_bins:	
					#Fourier transform of top-hat window function SQUARED
					W2 = ((3.0/(self.kh_bins*R_i)**2) * (np.sin(self.kh_bins*R_i)/(self.kh_bins*R_i) - np.cos(self.kh_bins*R_i)))**2
					W2_next = W2[1:]
					integral = 0.5 * delta_k * (k2_Pk[:-1]*W2[:-1] + k2_Pk_next*W2_next)	# [:-1] to ignore last point
					I += [integral.sum()]		

				I = np.array(I)
				sigma = I / (2*np.pi**2)

				Mass_t = rhom*(4.0*np.pi/3.0)*R_bins**3
				nu_t = delta_c/sigma
				# Columns: R_bins (Mpc/h)	Mass (Msun/h)	nu (adim)
				table = np.vstack((R_bins, Mass_t, nu_t)).T
				np.savetxt(table_name, table)

			# Load table ----------------------------------------------------------------------------------------
			table = np.loadtxt(table_name)
			Mass_t = table[:,1]
			nu_t = table[:,2]

			# Interpolation (cubic spline) ----------------------------------------------------------------------
			if ((M200_h<Mass_t[0]) or (M200_h>Mass_t[-1])):
				print 'WARNING: M200_h is out of range in '+table_name+'. The CubicSpline will extrapolate!'
			cs = CubicSpline(Mass_t,nu_t)
			nu = cs(M200_h)

			# Plot the interpolation
			if False:
				xM = np.geomspace(Mass_t[0], Mass_t[-1], 100)
				plt.figure()
				plt.plot(Mass_t,nu_t,'ko')
				plt.plot(xM, cs(xM), 'r-')
				plt.plot(M200_h,nu, 'bo')
				plt.title('M(nu) interpolation')
				plt.show()

			# Bias factor ---------------------------------------------------------------------------------------
			y = np.log10(Delta)
			A = 1.0 + 0.24*y*np.exp(-(4./y)**4)
			a = 0.44*y - 0.88
			B = 0.183
			b = 1.5
			C = 0.019 + 0.107*y + 0.19*np.exp(-(4.0/y)**4.0)
			c = 2.4	
			delta_c = 1.686
			bias = 1.0 - A * nu**a /(nu**a + delta_c**a) + B * nu**b + C * nu**c

			return bias


		# Computes bias factor
		bias = _bias_factor(M200_h, self.z, Delta)

		Dang_h = self.cosmo.angular_diameter_distance(self.z).value * self.cosmo.h 	# in Mpc/h

		rhoc = self.cosmo.critical_density(self.z)
		rhoc = rhoc.to(units.Msun/units.Mpc**3).value / self.cosmo.h**2			#critical density (h^2 Msun Mpc^-3)
		rhom = rhoc * self.cosmo.Om(self.z)		#mean density at z	

		# Constants of the integral (i.e. these factors are not functions of l)
		cte = rhom * bias / (2*np.pi * (1+self.z)**3 * Dang_h**2)

		# Computes matter power spectrum
		kh_0, Pk_0 = _power_spectrum_camb(self.z)

		# Interpolates for the actual bins we want
		cs = CubicSpline(kh_0,Pk_0)
		Pk = cs(self.kh_bins)	

		# Integrates
		#l_bins = self.kh_bins * (1+z)*Dang_h
		delta_l = self.l_bins[1:] - self.l_bins[:-1]
		F = self.l_bins * Pk
		F_next = F[1:]

		#print 'Shear integral...'
		I = []
		for i in xrange(len(r_h)):	
			# Computes bessel function
			#theta = r_i/Dang_h
			J2_i = self.J2[i,:] #sp.jn(2, l_bins*theta)
			J2_next = J2_i[1:]
			integral = 0.5 * delta_l * (F[:-1]*J2_i[:-1] + F_next*J2_next)	# [:-1] to ignore last point
			I += [integral.sum()]	

		I = np.asarray(I)
		second_halo = cte * I 		# Density contrast in h*Msun/Mpc2
		second_halo /= 1e6**2		# ...in h*Msun/pc2

		#table=np.vstack((r,second_halo)).T
		#np.savetxt('2ht_profile_'+sample+'.cat', table)

		return second_halo




class DensityModels:
	'''
	Module to handle density models and contruct composite models to fit any profile.
	'''

	def __init__(self, z, cosmo=None):
		self.z = z
		self.cosmo = cosmo
		self.density = Density(z, cosmo)

	def Bar(self, logMstar_0=13.,logMstar_fixed=False):
		model = models.custom_model(self.density.Bar)
		init_mod = model(logMstar_h=logMstar_0, name='Bar')
		# Constraints
		init_mod.logMstar_h.fixed = logMstar_fixed
		init_mod.logMstar_h.bounds = (7., 14.)
		return init_mod

	def Bar_with_M200(self, logM200_0=13., logM200_fixed=False):
		model = models.custom_model(self.density.Bar_with_M200)
		init_mod = model(logM200_h=logM200_0, name='Bar_with_M200')
		# Constraints
		init_mod.logM200_h.fixed = logM200_fixed
		init_mod.logM200_h.bounds = (10., 16.)
		return init_mod

	def SIS(self, disp_0=300., disp_fixed=False):
		model = models.custom_model(self.density.SIS)
		init_mod = model(disp=disp_0, name='SIS')
		# Constraints
		init_mod.disp.fixed = disp_fixed
		init_mod.disp.bounds = (0., 2000.)
		return init_mod

	def NFW(self, logM200_0=13., logM200_fixed=False):
		model = models.custom_model(self.density.NFW)
		init_mod = model(logM200_h=logM200_0, name='NFW')
		# Constraints
		init_mod.logM200_h.fixed = logM200_fixed
		init_mod.logM200_h.bounds = (10., 16.)
		return init_mod

	def NFWCombined(self, logM200_0=13., disp_offset_0=0.1, p_cen_0=1.,
		logM200_fixed=False, disp_offset_fixed=False, p_cen_fixed=False):

		model = models.custom_model(self.density.NFWCombined)
		init_mod = model(logM200_h=logM200_0, disp_offset_h=disp_offset_0, p_cen=p_cen_0, name='NFWCombined')
		# Constraints
		init_mod.logM200_h.fixed = logM200_fixed
		init_mod.disp_offset_h.fixed = disp_offset_fixed
		init_mod.p_cen.fixed = p_cen_fixed
		init_mod.logM200_h.bounds = (10., 16.)	
		init_mod.disp_offset_h.bounds = (0., 1.)
		init_mod.p_cen.bounds = (0., 1.)
		return init_mod		

	def SecondHalo(self, logM200_0=13., Delta=200.,
		logM200_fixed=False, Delta_fixed=True):
		if not Delta_fixed: print 'WARNING: You should fix Delta.'
		
		model = models.custom_model(self.density.SecondHalo)
		init_mod = model(logM200_h=logM200_0, Delta=Delta, name='SecondHalo')

		# Constraints
		init_mod.logM200_h.fixed = logM200_fixed
		init_mod.Delta.fixed = Delta_fixed		# Always fix Delta
		init_mod.logM200_h.bounds = (10., 16.)
		return init_mod	

	def AddModels(self, model_list):

		total_model = model_list[0]
		for m in model_list[1:]:
			total_model += m
		total_model.name = '+'.join(total_model.submodel_names)

		# Tie M200 for the compund model
		striped_param_names = [i[:-2] for i in total_model.param_names]
		striped_param_flags = ['logM200_h' in i for i in striped_param_names]
		num_M200 = sum(striped_param_flags)

		if num_M200 <= 1: return total_model

		first_M200 = np.where(striped_param_flags)[0][0]
		first_M200pos = int(total_model.param_names[first_M200][-1])
		tied_M200 = lambda M: getattr(total_model, total_model.param_names[first_M200])
		for i in range(num_M200):
			ii = i + first_M200pos + 1
			total_model.__dict__['_constraints']['tied']['logM200_h_'+str(ii)] = tied_M200
		return total_model




# Probamos...
'''
z=0.3
M200=2.e14
disp_off=0.2
p=0.7
r0 = np.geomspace(0.1,10.,20)

density = Density(Planck15)
if False:
	t0 = time.time()
	bcg=density.BCG_with_M200(r_h=r0,M200_h=M200)
	nfw_cen= density.NFW(r_h=r0, M200_h=M200) 
	nfw_off= density.NFWoff(r_h=r0, z=z, M200_h=M200, disp_offset_h=disp_off)
	nfw_2h=density.SecondHalo(r_h=r0, z=z, M200_h=M200, Delta=200)
	print 'Min: ',(time.time()-t0)/60.
	table=np.vstack((r0,bcg,nfw_cen,nfw_off,nfw_2h))
	np.savetxt('profiles.cat',table)
else:
	table=np.loadtxt('profiles.cat')
	r0  = table[0,:]
	bcg = table[1,:]
	nfw_cen = table[2,:]
	nfw_off = table[3,:]
	nfw_2h  = table[4,:]

densmodel=DensityModels(**cosmo)

bcg_model = densmodel.BCG_with_M200(M200_0=1.e10)
nfw_cen_model = densmodel.NFW(M200_0=1.e10)
nfw_model = densmodel.NFWCombined(z_fix=z, M200_0=1.e10, disp_offset_0=0.1, p_cen_0=0.9)
nfw2h_model = densmodel.SecondHalo(z_fix=z, M200_0=1.e10)

shear = bcg + p*nfw_cen + (1-p)*nfw_off + nfw_2h
total_model = bcg_model + nfw_model + nfw2h_model

tied_M200 = lambda M: total_model.M200_1
total_model.M200_0.tied = tied_M200
total_model.M200_2.tied = tied_M200

fitter = fitting.LevMarLSQFitter()
out= fitter(total_model,r0,shear,maxiter=100,acc=1e-2, epsilon=1.e-2)
#out= fitter(total_model,r0,shear,maxiter=10000,acc=1e-16)



from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
DLS=np.array(cosmo.angular_diameter_distance_z1z2(Zl,Zs))
dl = np.array(cosmo.angular_diameter_distance(zl))
'''