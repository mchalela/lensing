import os
import time
import numpy as np
import matplotlib.pyplot as plt ; plt.ion()
import multiprocessing 
from contextlib import closing

import emcee
#import corner
from astropy import units
from astropy.cosmology import Planck15 #, LambdaCDM
from lensing.densmodel import Density

DM = Density(Planck15)
###############################################################################
#--------------------------------------------------------------------
# probability of the data given the model
def lnlike(theta, z, rbins, data, stddev):
    logm, p, offset = theta
    #logm, offsets = theta
    
    # calculate the model

    bcg_model = DM.BCG_with_M200(r_h=rbins, M200_h=10**logm)
    nfw_model = DM.NFW(r_h=rbins, z=z, M200_h=10**logm)
    nfwoff_model = DM.NFWoff(r_h=rbins, z=z, M200_h=10**logm, disp_offset_h=offset)
    shalo_model = DM.SecondHalo(r_h=rbins, z=z, M200_h=10**logm)

    model = bcg_model + p*nfw_model + (1-p)*nfwoff_model + shalo_model

    
    diff = data - model
    lnlikelihood = -0.5 * np.sum(diff**2 / stddev**2)
    return lnlikelihood

# uninformative prior
def lnprior(theta):
    logm, p, offset = theta
    #if 10 < logm < 16 and 0.0 <= offset < 5.0:
    #    return 0.0
    #else:
    #    return -np.inf
    #process = multiprocessing.current_process()
    #print process.pid
    #print ' Model: logm = ',logm,'	-	p = ',p,'	-	offset = ',offset
    if 12 < logm < 16 and 0.0 <= offset < 1. and 0.0 <= p <= 1.:
        return 0.0
    else:
        return -np.inf

# posterior probability
def lnprob(theta, z, rbins, data, stddev):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, z, rbins, data, stddev)


########################################################################################
# probability of the data given the model
# Fixing p and offset
def lnlike_fix_poff(theta, p, offset, z, rbins, data, stddev):
    logm = theta
    #logm, offsets = theta
    
    # calculate the model
    nfw_model = DM.NFW(r_h=rbins, z=z, M200_h=10**logm)
    #model = nfw_model
    nfwoff_model = DM.NFWoff(r_h=rbins, z=z, M200_h=10**logm, disp_offset_h=offset)
    model = p*nfw_model + (1-p)*nfwoff_model
    
    #c = ClusterEnsemble(z)
    #c.m200 = [10 ** logm]
    #c.calc_nfw(rbins=rbins, offsets=[offsets])
    #model = c.deltasigma_nfw.mean(axis=0).value
    
    diff = data - model
    lnlikelihood = -0.5 * np.sum(diff**2 / stddev**2)
    return lnlikelihood

# uninformative prior
def lnprior_fix_poff(theta):
    logm = theta
    #if 10 < logm < 16 and 0.0 <= offset < 5.0:
    #    return 0.0
    #else:
    #    return -np.inf
    process = multiprocessing.current_process()
    print process.pid
    #print ' Model: logm = ',logm,'	-	p = ',p,'	-	offset = ',offset
    if 12. < logm < 16.:
        return 0.0
    else:
        return -np.inf

# posterior probability
def lnprob_fix_poff(theta, p, offset, z, rbins, data, stddev):
    lp = lnprior_fix_poff(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_fix_poff(theta, p, offset, z, rbins, data, stddev)


########################################################################################
# probability of the data given the model
# Fixing offset
def lnlike_fix_off(theta, offset, z, rbins, data, stddev):
    logm, p = theta
    #logm, offsets = theta
    
    # calculate the model
    nfw_model = DM.NFW(r_h=rbins, z=z, M200_h=10**logm)
    #model = nfw_model
    nfwoff_model = DM.NFWoff(r_h=rbins, z=z, M200_h=10**logm, disp_offset_h=offset)
    model = p*nfw_model + (1-p)*nfwoff_model
    
    #c = ClusterEnsemble(z)
    #c.m200 = [10 ** logm]
    #c.calc_nfw(rbins=rbins, offsets=[offsets])
    #model = c.deltasigma_nfw.mean(axis=0).value
    
    diff = data - model
    lnlikelihood = -0.5 * np.sum(diff**2 / stddev**2)
    return lnlikelihood

# uninformative prior
def lnprior_fix_off(theta):
    logm, p = theta
    #if 10 < logm < 16 and 0.0 <= offset < 5.0:
    #    return 0.0
    #else:
    #    return -np.inf
    process = multiprocessing.current_process()
    print process.pid
    #print ' Model: logm = ',logm,'	-	p = ',p,'	-	offset = ',offset
    if 12. < logm < 16. and 0.0 < p <=1.:
        return 0.0
    else:
        return -np.inf

# posterior probability
def lnprob_fix_off(theta, offset, z, rbins, data, stddev):
    lp = lnprior_fix_off(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_fix_off(theta, offset, z, rbins, data, stddev)

#####################################################################
#--------------------------------------------------------------------
def mcmc_create_samples(args, ndim=3, nwalkers=10, steps=200, file_name='default'):

	#Sample the posterior using emcee
	#ndim = 
	#nwalkers = 10
	#steps = 300
	samples_file = 'samples_dim.'+str(nwalkers)+'.'+str(steps)+'.'+str(ndim)+'.'+file_name+'.txt' 

	p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
	p0[:,0] = p0[:,0] + 14.  # start somewhere close to true logm ~ 14
	#p0[:,0] = p0[:,0] + 13.5  # start somewhere close to true logm ~ 14

	if ndim == 2:
		lnprob_fn = lnprob_fix_off
	elif ndim == 3:
		lnprob_fn = lnprob
	
	with closing(multiprocessing.Pool()) as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_fn, args=args, pool=pool)
		# the MCMC chains take some time: about 49 minutes for the 500 samples below
		t0 = time.time()
		pos, prob, state = sampler.run_mcmc(p0, steps)
		print 'Tiempo: ', (time.time()-t0)/60., '   ', file_name
		samples = sampler.chain.reshape((-1, ndim))
		pool.terminate()

	# save the chain for later
	np.savetxt(samples_file, samples)

	return None



##########################################################################
'''
# TOY MODEL
# Generate true data and adds noise
logm_true = 14.5
offset_true = 0.3
p_true = 0.7
nbins = 20
z = 0.2
rbins = np.geomspace(0.1,30.,nbins)
name='toy'

DM = Density(Planck15)
bcg_model = DM.BCG_with_M200(r_h=rbins, M200_h=10**logm_true)
nfw_model = DM.NFW(r_h=rbins, z=z, M200_h=10**logm_true)
nfwoff_model = DM.NFWoff(r_h=rbins, z=z, M200_h=10**logm_true, disp_offset_h=offset_true)
shalo_model = DM.SecondHalo(r_h=rbins, z=z, M200_h=10**logm_true)

model_true = bcg_model + p_true*nfw_model + (1-p_true)*nfwoff_model + shalo_model

# add scatter with a stddev of 20% of data
noise = np.random.normal(scale=model_true*0.2, size=nbins)
model_obs = model_true + noise
yerr = np.abs(model_true/3)  # 33% error bars

plt.plot(rbins,model_true,'--') 
plt.plot(rbins,model_obs,'.') 
plt.errorbar(rbins, model_obs, yerr=yerr, fmt='None', ecolor='k')
'''


###########################################################################
# READ DATA
'''
offset = 0.4	# in Mpc/h
name = 'bin4_CS82'
z=0.23
profile = np.loadtxt('../redMapper_test/profile'+name+'.cat')
rbins = profile[:,0] * 0.7		# R in Mpc/h
shear_obs = profile[:,1] / 0.7	# Density contrast in h*Msun/pc2
shear_err = profile[:,2] / 0.7
args_fix_off=(offset, z, rbins, shear_obs, shear_err)
args_fix_none=(z, rbins, shear_obs, shear_err)
mcmc_create_samples(args=args_fix_off, ndim=2, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
mcmc_create_samples(args=args_fix_none, ndim=3, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.3.'+name+'.txt'

name = 'bin4_KiDS'
z=0.23
profile = np.loadtxt('perfiles/profile'+name+'.cat')
rbins = profile[:,0] * 0.7		# R in Mpc/h
shear_obs = profile[:,1] / 0.7	# Density contrast in h*Msun/pc2
shear_err = profile[:,2] / 0.7
args_fix_off=(offset, z, rbins, shear_obs, shear_err)
args_fix_none=(z, rbins, shear_obs, shear_err)
mcmc_create_samples(args=args_fix_off, ndim=2, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
mcmc_create_samples(args=args_fix_none, ndim=3, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.3.'+name+'.txt'

name = 'bin4_combined'
z=0.23
profile = np.loadtxt('perfiles/profile'+name+'.cat')
rbins = profile[:,0] * 0.7		# R in Mpc/h
shear_obs = profile[:,1] / 0.7	# Density contrast in h*Msun/pc2
shear_err = profile[:,2] / 0.7
args_fix_off=(offset, z, rbins, shear_obs, shear_err)
args_fix_none=(z, rbins, shear_obs, shear_err)
mcmc_create_samples(args=args_fix_off, ndim=2, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.2.'+name+'.txt'
mcmc_create_samples(args=args_fix_none, ndim=3, nwalkers=10, steps=300, file_name=name)
samples_file = 'samples_dim.10.300.3.'+name+'.txt'

plt.plot(rbins,shear_obs,'.') 
plt.errorbar(rbins, shear_obs, yerr=shear_err, fmt='None', ecolor='k')
plt.xlim([3e-2,1e1])
plt.ylim([1e0,1e3])
plt.loglog()
p = 0.71
z = 0.249 #0.23



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

    axes[2].plot(sampler_chain[:, :, 2].T, color="k", alpha=0.4)
    #axes[2].axhline(offset_true, color="g", lw=2)
    axes[2].set_ylabel("offset")
    axes[2].set_xlabel("step number")    

if i_can_wait:
	burn_in_step = 50  # based on a rough look at the walker positions above
	samples = sampler_chain[:, burn_in_step:, :].reshape((-1, ndim))
	# save the chain for later
	np.savetxt('samples.txt', samples)
else:
	# read in a previously generated chain
	samples = np.loadtxt('samples.txt')


fig = corner.corner(samples,
                    labels=["$\mathrm{log}M_{200}$", "$p$", "$\sigma_\mathrm{off}$"],
                    truths=[logm_true, p_true, offset_true])
#fig.savefig('cornerplot.png')

fig = corner.corner(samples,
                    labels=["$\mathrm{log}M_{200}$"],
                    truths=[logm_true])
#fig.savefig('cornerplot.png')

samples[:, 0] = 10**(samples[:, 0])
M_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))



# Plot fitted profile for ndim = 2 (offset fixed)
M_fit = M_mcmc[0][0] ; M_perr = M_mcmc[0][1] ; M_merr = M_mcmc[0][2]
p_fit = M_mcmc[1][0] ; p_perr = M_mcmc[1][1] ; p_merr = M_mcmc[1][2]
offset_fit = offset

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



#plt.savefig('perfiles/'+name+'_fixed_offset.png', format='png',bbox_inches='tight')
#plt.savefig('perfiles/'+name+'.png', format='png',bbox_inches='tight')




#################################

# Plot fitted profile for ndim = 3
M_fit = M_mcmc[0][0] ; M_perr = M_mcmc[0][1] ; M_merr = M_mcmc[0][2]
p_fit = M_mcmc[1][0] ; p_perr = M_mcmc[1][1] ; p_merr = M_mcmc[1][2]
offset_fit = M_mcmc[2][0] ; offset_perr = M_mcmc[2][1] ; offset_merr = M_mcmc[1][2]



bcg_fit = DM2.BCG_with_M200(r_h=rbins, M200_h=M_fit)
nfw_fit = DM2.NFW(r_h=rbins, z=z, M200_h=M_fit)
nfwoff_fit = DM2.NFWoff(r_h=rbins, z=z, M200_h=M_fit, disp_offset_h=offset_fit)
shalo_fit = DM2.SecondHalo(r_h=rbins, z=z, M200_h=M_fit)

model_fit = bcg_fit + p_fit*nfw_fit + (1-p_fit)*nfwoff_fit + shalo_fit

M_label = '$M_{200} = $'+str('%.2f' % (M_fit/1e14))+' $^{'+str('%.2f' % (M_perr/1e14))+'} ' \
			+' _{'+str('%.2f' % (M_merr/1e14))+'}$ '+' $[10^{14}$ '+'$M_{\odot} h^{-1}]$'

M_label += ' ; $p = $'+str('%.2f' % p_fit)+' $^{'+str('%.2f' % p_perr)+'} ' \
			+' _{'+str('%.2f' % p_merr)+'}$ '

M_label += ' ; $\sigma_{off} = $'+str('%.2f' % offset_fit)+' $^{'+str('%.2f' % offset_perr)+'} ' \
			+' _{'+str('%.2f' % offset_merr)+'}$ '+'$[Mpc \, h^{-1}]$'

plt.plot(rbins,model_obs,'.')#, label=name) 
plt.errorbar(rbins, model_obs, yerr=yerr, fmt='None', ecolor='k')
plt.plot(rbins,model_true,'k-',alpha=0.6, label='True')
plt.plot(rbins,model_fit,'b-', label='Fit') 
plt.plot(rbins,bcg_fit,'--',alpha=0.7)
plt.plot(rbins,p_fit*nfw_fit,'--',alpha=0.7)
plt.plot(rbins,(1-p_fit)*nfwoff_fit,'--',alpha=0.7) 
plt.plot(rbins,shalo_fit,'--',alpha=0.7)
plt.xlim([0.8e-1,3.5e1])
plt.ylim([1e0,1e3])
plt.loglog()
plt.legend()
plt.title(M_label)
plt.xlabel('r [Mpc$\,h^{-1}$]')
plt.ylabel(u'$\Delta\Sigma [h\,M_{\odot}\,pc^{-2}]$')



#plt.savefig('perfiles/'+name+'_fixed_offset.png', format='png',bbox_inches='tight')
#plt.savefig('perfiles/'+name+'.png', format='png',bbox_inches='tight')

'''