import torch
from torch import Tensor
from typing import Union
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils.probability.utils import (ndtr as Phi, phi)
from botorch.models.model import Model
import logging

class AugmentedExpectedImprovement(AnalyticAcquisitionFunction):
    def __init__(self, 
                model: Model,
                best_f: Union[float, Tensor],
                maximize: bool = True,
                **kwargs,
        ):
        super().__init__(model)
        self.best_f = best_f
        self.maximize = maximize
        
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei = sigma * _ei_helper(u)
        
        noise_level = self.model.likelihood.noise_covar.noise
        aei = ei * (1 - (noise_level / torch.sqrt(sigma**2 + noise_level**2)))
        
        return aei
    
def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)

def get_best_fx(model: Model, xs:Tensor, alpha: float=1.0, effective: bool = False):
    '''
        model : Model
        x : torch.Tensor
        effective : bool
        
        If effective is True, get the effective best solution.
            ref) Huang, Deng, et al. "Global optimization of stochastic black-box systems via sequential kriging meta-models." Journal of global optimization 34 (2006): 441-466.
    '''
    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(xs)
    mean = posterior.mean
    sigma = posterior.variance.sqrt()
    # sign = np.random.choice([-1, 1])
    # alpha *= sign
    
    return mean.max() if effective else (mean + sigma * alpha).max()

def get_best_x(model: Model, xs: Tensor, fxs: Tensor=None, alpha: float=1.0, noisy: bool = True, effective: bool = False):
# def get_best_x(model: Model, xs: Tensor, fxs: Tensor, alpha: float=1.0, noisy: bool = True, effective: bool = False, maximize: bool = True):
    '''
        model : Model
        x : torch.Tensor
        effective : bool
        
        If effective is True, get the effective best solution.
            ref) Huang, Deng, et al. "Global optimization of stochastic black-box systems via sequential kriging meta-models." Journal of global optimization 34 (2006): 441-466.
    '''
    
    # maximize = True if effective else False
    
    if noisy or effective:
        model.eval()
        model.likelihood.eval()
        posterior = model.posterior(xs)    
        mean = posterior.mean
        sigma = posterior.variance.sqrt()
        
        fxs = mean + sigma * alpha if effective else mean
        
        x_center = torch.clone(xs[fxs.argmax(), :]).detach()
        
        # if maximize:
        #     x_center = torch.clone(xs[fxs.argmax(), :]).detach()
        # else:
        #     x_center = torch.clone(xs[fxs.argmin(), :]).detach()
        
    else:
        x_center = torch.clone(xs[fxs.argmin(), :]).detach()
        
        # if maximize:
        #     x_center = torch.clone(xs[fxs.argmax(), :]).detach()
        # else:
        #     x_center = torch.clone(xs[fxs.argmin(), :]).detach()
    
    return x_center