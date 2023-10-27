import warnings
from typing import Union, Optional
from dataclasses import dataclass, field, InitVar

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator

from .docstrings import add_dataclass_docstring

@add_dataclass_docstring
@dataclass
class Design(LinearOperator):

    """
    Linear map representing multiply with [1,X] and its transpose.
    """

    X: Union[np.ndarray, scipy.sparse.csc_array]
    weights: Optional[np.ndarray] = None
    dtype: np.dtype = float
    standardize: InitVar[bool] = False
    intercept: InitVar[bool] = True

    def __post_init__(self, standardize, intercept):

        self.shape = (self.X.shape[0], self.X.shape[1]+1)
        n = self.shape[0]

        if self.weights is None:
            self.weights = np.ones(n)
        
        X, weights = self.X, self.weights

        # if standardizing, then the effective matrix is
        # xs = scaling_
        # (X - (1'W)^{-1} 1 W'X) S^{-1}
        # (X - 1 xm') @ diag(1/xs) = XS^{-1} - 1 xm/xs'

        sum_w = weights.sum()
        self.centers_ = X.T @ weights / sum_w

        if standardize:
            xm2 = (X*X).T @ weights / sum_w
            self.scaling_ = np.sqrt(xm2 - self.centers_**2)
        else:
            self.scaling_ = np.ones(self.shape[1]-1) # -1 for intercept
            
        if not intercept:
            self.centers_ *= 0

        if scipy.sparse.issparse(self.X):
            self.X = self.X.tocsc()
        else:
            self.X = (self.X - self.centers_[None,:]) / self.scaling_[None,:]
            self.X = np.asfortranarray(self.X)

    # LinearOperator API

    def _matvec(self, x):
        intercept = x[0]
        coef = x[1:]
        
        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.centers_, self.scaling_
            if coef.ndim == 1:
                coef = coef / xs
                prod_ = X @ coef - np.sum(coef * xm) + intercept
            else:
                coef = coef / xs[:,None]
                prod1 = xm @ coef
                prod_ = X @ coef - prod1[None,:] + intercept
        else:
            prod_ = X @ coef + intercept
        return prod_

    def _rmatvec(self, r):

        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.centers_, self.scaling_
            if r.ndim == 1:
                XtR = (X.T @ r - np.sum(r) * xm) / xs
            else:
                val1 = X.T @ r
                val2 = np.multiply.outer(xm, r.sum(0))
                XtR = (val1 - val2) / xs[:,None]
        else:
            XtR = X.T @ r
        if r.ndim == 1:
            return np.hstack([r.sum(0), XtR])
        else:
            return np.concatenate([r.sum(0).reshape((-1,1)), XtR], axis=0)

    def quadratic_form(self,
                       G=None,
                       columns=None):
        '''
        A'GA[:,E]

        where A is the effective matrix

        [1, XS^{-1} - 1 (xm/xs)']

        and E is a subset of columns
        '''

        # A is effective matrix
        # A = XS^{-1} - 1 (xm/xs)' = (X - 1 xm')S^{-1}
        # GA = G(X - 1 xm')S^{-1}
        # A'G1 = S^{-1}(X' - xm 1')G1 = S^{-1}X'G1 - S^{-1}xm 1'G1
        # A'GA = S^{-1}X'GXS^{-1} - S^{-1}X'G1 xm/xs' - xm/xs 1'GXS^{-1} + xm/xs 1'G1 (xm/xs)'
        
        n, p = self.shape[0], self.shape[1] - 1
        
        if columns is None:
            X_R = self.X
            columns = slice(0, p)
        else:
            X_R = self.X[:,columns]

        if G is None:
            XX_block = self.X.T @ X_R # have to assume this is not too expensive
            X1_block = self.X.sum(0) # X'1
            G_sum = n
            
        else:
            GX = G @ X_R
            G1 = G @ np.ones(G.shape[0])
            XX_block = self.X.T @ GX
            X1_block = self.X.T @ G1
            G_sum = G1.sum()
            
        if scipy.sparse.issparse(XX_block):
            XX_block = XX_block.toarray()
            
        # correct XX_block for standardize
        
        xm, xs = self.centers_, self.scaling_

        if columns is not None:
            XX_block -= (np.multiply.outer(X1_block, xm[columns]) + np.multiply.outer(xm, X1_block[columns]))
            XX_block += np.multiply.outer(xm, xm[columns]) * G_sum
            XX_block /= np.multiply.outer(xs, xs[columns])
        else:
            XX_block -= (np.multiply.outer(X1_block, xm) + np.multiply.outer(xm, X1_block))
            XX_block += np.multiply.outer(xm, xm) * G_sum
            XX_block /= np.multiply.outer(xs, xs)

        X1_block -= G_sum * xm
        X1_block /= xs
        
        Q = np.zeros((XX_block.shape[0] + 1,
                      XX_block.shape[1] + 1))
        Q[1:,1:] = XX_block
        Q[1:,0] = X1_block
        if columns is not None:
            Q[0,1:] = X1_block[columns]
        else:
            Q[0,1:] = X1_block
        Q[0,0] = G_sum
        
        return Q

def _get_design(X,
                sample_weight,
                standardize=False,
                intercept=True):
    if isinstance(X, Design):
        return X
    else:
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        return Design(X,
                      sample_weight,
                      standardize=standardize,
                      intercept=intercept)

@add_dataclass_docstring
@dataclass
class Base(object):
    
    X: Union[np.ndarray, scipy.sparse.csc_array, Design]
    y : np.ndarray

@add_dataclass_docstring
@dataclass
class Penalty(object):
    
    lambda_val : float
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None

    def penalty(self, coef):

        if np.any(coef < self.lower_limits - 1e-3) or np.any(self > self.upper_limits + 1e-3):
            return np.inf
        val = self.lambda_val * (
            self.alpha * (self.penalty_factor * np.fabs(coef)).sum() + 
            (1 - self.alpha) * np.linalg.norm(coef)**2)
        return val
