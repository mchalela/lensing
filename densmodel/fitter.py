from abc import ABCMeta
import abc
from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling import models



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
        print logM200
        if 10 < logM200 < 16: # and 0.0 <= offset < 1. and 0.0 <= p <= 1.:
            return 0.0 #1./(16.-10.)
        else:
            return lmin
    
    def offset_prior(offset):
        if 0.0 <= offset < 1.: # and 0.0 <= p <= 1.:
            return np.log(1./(1.-0.))
        else:
            return lmin       

    def p_cen_prior(p_cen):
        if 0.0 <= p_cen <= 1.:
            return np.log(1./(1.-0.))
        else:
            return lmin  

    def logMstar_prior(logMstar):
        print logMstar
        if 7 < logMstar < 14:
            return 0.0 #1./(14.-7.)
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
        self.x = x
        self.y = y
        
        self.model = model
        
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
        
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
        
    def evaluate(self, pars):
        _fitter_to_model_params(self.model, pars)
        
        mean_model = self.model(self.x)
        
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

        self.x = x
        self.y = y
        self.model = model

    @abc.abstractmethod
    def loglikelihood(self, parameters):
        pass
    
    @abc.abstractmethod        
    def logprior(self, parameters):
        pass
    
    def logposterior(self, parameters):
        lp = self.logprior(parameters)
        lll= self.loglikelihood(parameters)
        print '--------------', lp, lll
        return lp + lll
        #return self.logprior(parameters) + self.loglikelihood(parameters)
    
    def __call__(self, parameters):
        return self.logposterior(parameters)


class GaussianLogPosterior(LogPosterior, object):
    
    def __init__(self, x, y, yerr, model):
        
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
    def logprior(self, pars):
        """
        Some hard-coded priors.
        
        """
        p = 1.
        for param, name in zip(pars, self.model.param_names):
            if name[-1] in ['0','1','2','3']: name = name[:-2] 
            p += ParamPrior(param, name)
        return p

    def loglikelihood(self, pars):
        _fitter_to_model_params(self.model, pars)
        
        mean_model = self.model(self.x)
        
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))

        return loglike
