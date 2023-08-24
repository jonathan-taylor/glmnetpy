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
    standardize: bool = False
    
    def __post_init__(self):

        n, p = self.shape = self.X.shape
        X, weights = self.X, self.weights

        if self.standardize:
            sum_w = weights.sum()
            self.xm = X.T @ weights / sum_w
            xm2 = (X*X).T @ weights / sum_w
            self.xs = np.sqrt(xm2 - self.xm**2)
        else:
            self.xm = np.zeros(self.shape[1])
            self.xs = np.ones(self.shape[1])

        if scipy.sparse.issparse(self.X):
            self.X = self.X.tocsc()
        else:
            self.X = self.X - np.multiply.outer(np.ones(n), self.xm)
            self.X = self.X / self.xs[None,:]
            self.X = np.asfortranarray(self.X)


        
    # the map presumes X has column of 1's appended
    
    def linear_map(self,
                   coef,
                   intercept=0):
    
        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.xm, self.xs
            if coef.ndim == 1:
                coef = coef / xs
                eta = X @ coef - np.sum(coef * xm) + intercept
            else:
                coef = coef / xs[:,None]
                prod1 = xm @ coef
                eta = X @ coef - prod1[None,:] + intercept
        else:
            eta = X @ coef + intercept
        return eta

    def adjoint_map(self,
                    r):
   
        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.xm, self.xs
            if r.ndim == 1:
                V1 = (X.T @ r - np.sum(r) * xm) / xs
            else:
                val1 = X.T @ r
                val2 = np.multiply.outer(xm, r.sum(0))
                V1 = (val1 - val2) / xs[:,None]
            return V1, r.sum(0)
        else:
            return X.T @ r, r.sum(0) 

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
    penalty_factor: Optional[Union[float, np.ndarray]] = None

@add_dataclass_docstring
@dataclass
class Options(object):
    
    intercept: bool = True
    exclude: list = field(default_factory=list)
    standardize: bool = False
    
