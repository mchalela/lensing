import os
import time
import numpy as np
#import matplotlib.pyplot as plt ; plt.ion()
#import multiprocessing

import emcee
#import corner
from astropy import units
from astropy.cosmology import Planck15 #, LambdaCDM
from lensing import shear, densmodel




name = 'bin4_COMB'
z=0.25
offset = 0.4	# in Mpc/h
path = '/home/martin/Documentos/Doctorado/Lentes/lensing/redMapper_test/redMapper_COMB.profile'
#path = '/home/mchalela/redMaPPer/redMapper_COMB.profile'
profile = shear.Profile.read_profile(path)
rbins = profile['r_hMpc']		# R in Mpc/h
shear_obs = profile['shear'] / 0.7	# Density contrast in h*Msun/pc2
shear_err = profile['shear_error'] / 0.7
args_fix_off=(offset, z, rbins, shear_obs, shear_err)
args_fix_none=(z, rbins, shear_obs, shear_err)
print 'Ndim 2'
densmodel.mcmc.mcmc_create_samples(args=args_fix_off, ndim=2, nwalkers=10, steps=300, file_name=name, threads=4)
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
print 'Ndim 3'
densmodel.mcmc.mcmc_create_samples(args=args_fix_none, ndim=3, nwalkers=10, steps=300, file_name=name, threads=56)
samples_file = 'samples_dim.10.300.3.'+name+'.txt'


p=0.7
DM = densmodel.Density(Planck15)
nfw_model = DM.NFW(r_h=rbins, z=z, M200_h=2.5*10**14.)
nfwoff_model = DM.NFWoff(r_h=rbins, z=z, M200_h=2.5*10**14., disp_offset_h=0.4)
model = p*nfw_model + (1-p)*nfwoff_model

plt.plot(rbins,model,'k.-')
plt.plot(rbins,p*nfw_model,'r--',alpha=0.7)
plt.plot(rbins,(1-p)*nfwoff_model,'r--',alpha=0.7)
plt.plot(rbins, shear_obs,'bx')


i_can_wait = True
nwalkers=10
steps=300
ndim=2
# read in a previously generated chain
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
samplesf = np.loadtxt(samples_file)
sampler_chain = samplesf.reshape((nwalkers,steps,ndim))

if i_can_wait:
	fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
	axes[0].plot(sampler_chain[:, :, 0].T, color="k", alpha=0.4)
	#axes[0].axhline(logm_true, color="g", lw=2)
	axes[0].set_ylabel("log-mass")

	axes[1].plot(sampler_chain[:, :, 1].T, color="k", alpha=0.4)
	#axes[1].axhline(p_true, color="g", lw=2)
	axes[1].set_ylabel("p")
	axes[1].set_xlabel("step number")

	#axes[2].plot(sampler_chain[:, :, 2].T, color="k", alpha=0.4)
	#axes[2].axhline(offset_true, color="g", lw=2)
	#axes[2].set_ylabel("offset")
	#axes[2].set_xlabel("step number")    

	burn_in_step = 50  # based on a rough look at the walker positions above
	log_samples = sampler_chain[:, burn_in_step:, :].reshape((-1, ndim))

samples[:, 0] = 10**(log_samples[:, 0])

M_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
# Plot fitted profile for ndim = 2 (offset fixed)
M_fit = M_mcmc[0][0] ; M_perr = M_mcmc[0][1] ; M_merr = M_mcmc[0][2]
p_fit = M_mcmc[1][0] ; p_perr = M_mcmc[1][1] ; p_merr = M_mcmc[1][2]
offset_fit = offset

fig = corner.corner(log_samples,
                    labels=["$\mathrm{log}M_{200}$", '$p$'],
                    truths=[np.log10(M_fit), p_fit])

fig = corner.corner(samples,
                    labels=["$\mathrm{log}M_{200}$", "$p$", "$\sigma_\mathrm{off}$"],
                    truths=[logm_true, p_true, offset_true])

nfw_fit = DM.NFW(r_h=rbins, z=z, M200_h=M_fit)
nfwoff_fit = DM.NFWoff(r_h=rbins, z=z, M200_h=M_fit, disp_offset_h=offset_fit)
model_fit = p_fit*nfw_fit + (1-p_fit)*nfwoff_fit

M_label = '$M_{200} = $'+str('%.2f' % (M_fit/1e14))+' $^{'+str('%.2f' % (M_perr/1e14))+'} ' \
			+' _{'+str('%.2f' % (M_merr/1e14))+'}$ '+' $[10^{14}$ '+'$M_{\odot} h^{-1}]$'
M_label += ' ; $p = $'+str('%.2f' % p_fit)+' $^{'+str('%.2f' % p_perr)+'} ' \
			+' _{'+str('%.2f' % p_merr)+'}$ '
M_label += ' ; (fixed) $\sigma_{off} = $'+str('%.2f' % offset_fit)+' $[Mpc \, h^{-1}]$'

#plt.plot(rbins,model_true,'--') 
plt.plot(rbins,shear_obs,'.', label=name) 
plt.errorbar(rbins, shear_obs, yerr=shear_err, fmt='None', ecolor='k')
plt.plot(rbins,model_fit,'-') 
plt.plot(rbins,p_fit*nfw_fit,'--',alpha=0.7)
plt.plot(rbins,(1-p_fit)*nfwoff_fit,'--',alpha=0.7) 
plt.xlim([3e-2,1e1])
plt.ylim([1e0,1e3])
plt.loglog()
plt.legend()
plt.title(M_label)
plt.xlabel('r [Mpc$\,h^{-1}$]')
plt.ylabel(u'$\Delta\Sigma [h\,M_{\odot}\,pc^{-2}]$')