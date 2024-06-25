import torch
from torch import Tensor
from typing import Union
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils.probability.utils import (ndtr as Phi, phi)
from botorch.models.model import Model

class AugmentedExpectedImprovement(AnalyticAcquisitionFunction):
    def __init__(self, 
                model: Model,
                best_f: Union[float, Tensor],
                tau: float,
                maximize: bool = True,
                **kwargs,
        ):
        super().__init__(model)
        self.best_f = best_f
        self.tau = tau
        self.maximize = maximize
        
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei = sigma * _ei_helper(u)
        aei = ei * (1 - (self.tau/(torch.sqrt(sigma**2)+self.tau**2).sqrt()))
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