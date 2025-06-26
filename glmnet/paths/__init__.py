"""
The glmnet.paths submodule provides fast path solvers for penalized regression and classification models.
Includes estimators for Gaussian, binomial, Poisson, multinomial, and multi-response regression.
"""

# fast paths

from .lognet import LogNet
from .gaussnet import GaussNet
from .fishnet import FishNet
from .multigaussnet import MultiGaussNet
from .multiclassnet import MultiClassNet
