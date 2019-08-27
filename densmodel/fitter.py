from abc import ABCMeta
import abc
import time
import numpy as np
import scipy.optimize
import emcee
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params
from multiprocessing import Pool


# The problem with emcee is that the probability function and its arguments have to be pickable
# This is a very ugly solution, but it works... we dont send the model as an argument but 
# we can set it as a global variable
GlobalModel = None
isGlobalModelclear = True

def Chi2Reduced(model, data, err, df=1):
    '''
    Computes the reduced Chi2 for a given model and data
    '''
    nbin = len(data)
    chi2 = ((model-data)**2/err**2).sum() / float(nbin-1-df)
    return chi2

def ParamPrior(param, param_name):
    '''
    Choose the prior for a given parameter
    '''
    lmin = -1e10

    def logM200_prior(logM200):
        if 10 < logM200 < 16: 
            return 0.0
        else:
            return lmin
    
    def offset_prior(offset):
        if 0.0 <= offset < 1.: 
            return 0.0
        else:
            return lmin       

    def p_cen_prior(p_cen):
        if 0.0 <= p_cen <= 1.:
            return 0.0
        else:
            return lmin  

    def logMstar_prior(logMstar):
        if 7 < logMstar < 14:
            return 0.0 
        else:
            return lmin

    #print param_name, param
    if param_name == 'logMstar_h': return logMstar_prior(param)
    if param_name == 'logM200_h': return logM200_prior(param)
    if param_name == 'disp_offset_h': return offset_prior(param)
    if param_name == 'p_cen': return p_cen_prior(param)
    if param_name == 'Delta': return 0.0


class ObjectiveFunction(object):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def __call__(self):
        """
        Any objective function must have a `__call__` method that 
        takes parameters as a numpy-array and returns a value to be 
        optimized or sampled.
        
        """
        pass


class LogLikelihood(ObjectiveFunction):
    __metaclass__ = ABCMeta

    def __init__(self, x, y, model):
        """
        x : iterable
            x-coordinate of the data. Could be multi-dimensional.
        
        y : iterable
            y-coordinate of the data. Could be multi-dimensional.
        
        model: probably astropy.modeling.FittableModel instance
            Your model
        """
        global GlobalModel
        self.x = x
        self.y = y
        GlobalModel = model.copy()
        
    @abc.abstractmethod
    def evaluate(self, parameters):
        """
        This is where you define your log-likelihood. Do this!
        
        """
        pass
    
    @abc.abstractmethod
    def __call__(self, parameters):
        return self.loglikelihood(parameters)


class GaussianLogLikelihood(LogLikelihood, object):
    
    def __init__(self, x, y, yerr, model):
        """
        A Gaussian likelihood.
        
        Parameters
        ----------
        x : iterable
            x-coordinate of the data
            
        y : iterable
            y-coordinte of the data
        
        yerr: iterable
            the error on the data
            
        model: an Astropy Model instance
            The model to use in the likelihood.
        
        """
        global GlobalModel
        
        self.x = x
        self.y = y
        self.yerr = yerr
        GlobalModel = model.copy()
        
    def evaluate(self, pars):
        _fitter_to_model_params(GlobalModel, pars)
        mean_model = GlobalModel(self.x)
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))
        return loglike
    
    def __call__(self, pars):
        return self.evaluate(pars)        


class LogPosterior(ObjectiveFunction):
    __metaclass__ = ABCMeta
    
    def __init__(self, x, y, model):
        """
        x : iterable
            x-coordinate of the data. Could be multi-dimensional.
        
        y : iterable
            y-coordinate of the data. Could be multi-dimensional.
        
        model: probably astropy.modeling.FittableModel instance
            Your model
        """
        global GlobalModel

        self.x = x
        self.y = y
        GlobalModel = model.copy()

    @abc.abstractmethod
    def loglikelihood(self, parameters):
        pass
    
    @abc.abstractmethod        
    def logprior(self, parameters):
        pass
    
    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)
    
    def __call__(self, parameters):
        return self.logposterior(parameters)


class GaussianLogPosterior(LogPosterior, object):
    
    def __init__(self, x, y, yerr, model):
        global GlobalModel

        self.x = x
        self.y = y
        self.yerr = yerr
        GlobalModel = model.copy()
        
    def logprior(self, pars):
        """
        Some hard-coded priors.
        
        """
        p = 0.
        for param, name in zip(pars, GlobalModel.param_names):
            if name[-1] in ['0','1','2','3']: name = name[:-2] 
            p += ParamPrior(param, name)
        return p

    def loglikelihood(self, pars):
        _fitter_to_model_params(GlobalModel, pars)
        mean_model = GlobalModel(self.x)
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))
        return loglike


class Fitter(object):
    def __init__(self, r, shear, shear_err, model, start_params):
        '''
        Fitter method.
        Inputs:
        -------
          r:         ndarray. Distance to the lens centre in Mpc/h.
          shear:     ndarray. Density contrast in h*Msun/pc2.
          shear_err: ndarray. Density contrast error in h*Msun/pc2.
          model:     astropy model. Density model contructed with densmodel.DensityModels.
          start_params: list. Starting values for the fitter. If model is a compound
                model with multiple parameters, start_param values should be in
                the same order as given in model.param_names.

        Methods:
        --------
        clear:  Resets the model object. If you need to fit multiple models you
                should clear the fitter after each fit;
                i.e. instantiate the fitter, fit your model, clear the fitter, repeat.
        MCMC:   Uses emcee sampler to sample the Gaussian LogLikelihood (method='GLL')
                with no priors. If you want flat priors use method='GLP'. Usefull for
                expesive compound models.
        Minimize: Uses the scipy minimize method to minimize the negative Gaussian 
                LogLikelihood (method='GLL'). 

        '''

        global GlobalModel, isGlobalModelclear
        if not isGlobalModelclear:
            print ' Clearing previous model.'
            self.clear()

        self.r = r
        self.shear = shear
        self.shear_err = shear_err
        self.start = start_params
        GlobalModel = model.copy()
        isGlobalModelclear = False

    def clear(self):
        global GlobalModel, isGlobalModelclear

        GlobalModel = None
        isGlobalModelclear = True

    def MCMC(self, method='GLL', nwalkers=16, steps=300, sample_name='default'):

        if method == 'GLL':
            #loglike = log_likelihood
            loglike = GaussianLogLikelihood(self.r, self.shear, self.shear_err, GlobalModel)
        elif method == 'GLP':
            #loglike = log_probability
            loglike = GaussianLogPosterior(self.r, self.shear, self.shear_err, GlobalModel)

        #Sample the posterior using emcee
        ndim = len(GlobalModel.parameters)
        p0 = np.random.normal(self.start, ndim*[0.2], size=(nwalkers, ndim))

        pool = Pool()
        args = (self.r, self.shear, self.shear_err)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, pool=pool)
        # the MCMC chains take some time
        print 'Running MCMC...'
        t0 = time.time()
        pos, prob, state = sampler.run_mcmc(p0, steps)
        print 'Completed in {} min'.format((time.time()-t0)/60.)
        pool.terminate()

        # save the chain and return file name
        samples_file = sample_name+'.'+str(nwalkers)+'.'+str(steps)+'.'+str(ndim)+'.chain' 
        samples = sampler.chain.reshape((-1, ndim))
        np.savetxt(samples_file, samples)
        return sampler, samples_file


    def Minimize(self, method='GLL', verbose=True):

        if method == 'GLL':
            #loglike = log_likelihood
            loglike = GaussianLogLikelihood(self.r, self.shear, self.shear_err, GlobalModel)
        elif method == 'GLP':
            #loglike = log_probability
            loglike = GaussianLogPosterior(self.r, self.shear, self.shear_err, GlobalModel)
        neg_loglike = lambda x: -loglike(x)

        print 'Minimizing...'
        t0 = time.time()
        args = (self.r, self.shear, self.shear_err)
        output = scipy.optimize.minimize(neg_loglike, self.start, method="L-BFGS-B", tol=1.e-10)
        print 'Completed in {} min'.format((time.time()-t0)/60.)

        output = self.Minimize_OutputAnalysis(output)
        return output

    def Minimize_OutputAnalysis(self, output):

        nparam = len(GlobalModel.parameters)
        _fitter_to_model_params(GlobalModel, output.x)
        mean_model = GlobalModel(self.r)
        errors = np.diag(output.hess_inv.todense())**0.5
        chi2 = Chi2Reduced(mean_model, self.shear, self.shear_err, df=nparam)

        more = {'model_name': GlobalModel.name,
                'param_names': [n.encode('utf-8') for n in GlobalModel.param_names],
                'param_values': output.x,
                'param_errors': errors,
                'hess_inv': output.hess_inv,
                'jac': output.jac,
                'nit': output.nit,
                'chi2': chi2,
                'status': output.status,
                'success': output.success}
        output = scipy.optimize.optimize.OptimizeResult(more)
        return output

    def MCMC_OutputAnalysis(self, samples_file):
        pass


def Fitter2Model(model, params):
    '''
    Maps the parameters to the model so the best fit can be evaluated as model(r)
    '''
    _fitter_to_model_params(model, params)
    return None
