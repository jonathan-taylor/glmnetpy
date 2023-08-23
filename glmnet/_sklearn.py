from sklearn.base import BaseEstimator
from dataclasses import dataclass

from .base import Penalty, Options
from .glmnet_fit import GLMMixIn

@dataclass
class ElNetMixin(Options, Penalty):
    pass

@dataclass
class GLMNetMixin(GLMMixIn, Options, Penalty):
    pass

class ElNetEstimator(BaseEstimator, ElNetMixin):

    def fit(X, y, w):
        return 

class GLMNetEstimator(BaseEstimator, GLMNetMixin):

    def fit(X, y, w):
        return 
    
