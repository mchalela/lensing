from abc import ABCMeta
import abc
from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling import models



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
        return self.logprior(parameters) + self.loglikelihood(parameters)
    
    def __call__(self, parameters):
        return self.logposterior(parameters)


logmin = -10000000000000000.0

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
        # amplitude prior
        amplitude = pars[0]
        logamplitude = np.log(amplitude)
        
        logamplitude_min = -8.
        logamplitude_max = 8.0 
        p_amp = ((logamplitude_min <= logamplitude <= logamplitude_max) / \
                      (logamplitude_max-logamplitude_min))
        
        # mean prior
        mean = pars[1]
        
        mean_min = self.x[0]
        mean_max = self.x[-1]
        
        p_mean = ((mean_min <= mean <= mean_max) / (mean_max-mean_min))

        # width prior
        width = pars[2]
        logwidth = np.log(width)
        
        logwidth_min = -8.0
        logwidth_max = 8.0
        
        p_width = ((logwidth_min <= logwidth <= logwidth_max) / (logwidth_max-logwidth_min))

        pp = p_amp*p_mean*p_width
        
  
        if pp == 0 or np.isfinite(pp) is False:
            return logmin
        else:
            return np.log(pp)
        
    
    def loglikelihood(self, pars):
        _fitter_to_model_params(self.model, pars)
        
        mean_model = self.model(self.x)
        
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))

        return loglike