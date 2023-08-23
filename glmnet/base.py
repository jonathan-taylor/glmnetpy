from typing import Union, Optional
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse

from .docstrings import add_dataclass_docstring

@add_dataclass_docstring
@dataclass
class Design(object):

    X: Union[np.ndarray, scipy.sparse._csc.csc_array]
    weights: np.ndarray
    
    def __post_init__(self):
        if scipy.sparse.issparse(self.X):
            self.X = self.X.tocsc()
        else:
            self.X = np.asfortranarray(self.X)
        X, weights = self.X, self.weights
        sum_w = weights.sum()
        self.xm = X.T @ weights / sum_w
        xm2 = (X*X).T @ weights / sum_w
        self.xs = xm2 - self.xm**2

    # the map presumes X has column of 1's appended
    
    def linear_map(self,
                   beta,
                   a0):
    
        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.xm, self.xs
            beta = beta / xs
            eta = X @ beta - np.sum(beta * xm) + a0
        else:
            eta = X @ beta + a0
        return eta

    def wls_args(self):
        if not scipy.sparse.issparse(self.X):
            return {'x':self.X}
        else:
            return {'x_data_array':self.X.data,
                    'x_indices_array':self.X.indices,
                    'x_indptr_array':self.X.indptr,
                    'xm':self.xm,
                    'xs':self.xs}

@add_dataclass_docstring
@dataclass
class Base(object):
    
    X: Union[np.ndarray, scipy.sparse._csc.csc_array, Design]
    y : np.ndarray

@add_dataclass_docstring
@dataclass
class Penalty(object):
    
    lambda_val : float
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor = Optional[Union[float, np.ndarray]]

@add_dataclass_docstring
@dataclass
class Options(object):
    
    exclude: list = field(default_factory=list)
    weights: Optional[np.ndarray] = None
    intercept: bool = True
    
    
