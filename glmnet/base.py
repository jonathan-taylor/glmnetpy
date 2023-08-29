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

    X: Union[np.ndarray, scipy.sparse._csc.csc_array]
    weights: np.ndarray
    dtype: np.dtype = float
    standardize: InitVar[bool] = False
    intercept: InitVar[bool] = True

    
    def __post_init__(self, standardize, intercept):

        self.shape = (self.X.shape[0], self.X.shape[1]+1)
        n = self.shape[0]
        
        X, weights = self.X, self.weights

        # if standardizing, then the effective matrix is
        # xs = scaling_
        # (X - (1'W)^{-1} 1 W'X) S^{-1}
        # (X - 1 xm') @ diag(1/xs) = XS^{-1} - 1 xm/xs'

        if standardize:
            sum_w = weights.sum()
            self.centers_ = X.T @ weights / sum_w
            xm2 = (X*X).T @ weights / sum_w
            self.scaling_ = np.sqrt(xm2 - self.centers_**2)
        else:
            self.centers_ = np.zeros(self.shape[1]-1) # -1 for intercept
            self.scaling_ = np.ones(self.shape[1]-1)
            
        if not intercept:
            self.centers_ *= 0

        if scipy.sparse.issparse(self.X):
            self.X = self.X.tocsc()
        else:
            self.X = self.X - np.multiply.outer(np.ones(n), self.centers_)
            self.X = self.X / self.scaling_[None,:]
            self.X = np.asfortranarray(self.X)

    # LinearOperator API

    def _matvec(self, x):
        return self.linear_map(x[1:], x[0])

    def _rmatvec(self, y):
        r1, r2 = self.adjoint_map(y)
        return np.hstack([r2, r1])
    
    # the map presumes X has column of 1's appended
    
    def linear_map(self,
                   coef,
                   intercept=0):
    
        X = self.X
        if scipy.sparse.issparse(X):
            xm, xs = self.centers_, self.scaling_
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
            xm, xs = self.centers_, self.scaling_
            if r.ndim == 1:
                V1 = (X.T @ r - np.sum(r) * xm) / xs
            else:
                val1 = X.T @ r
                val2 = np.multiply.outer(xm, r.sum(0))
                V1 = (val1 - val2) / xs[:,None]
            return V1, r.sum(0)
        else:
            return X.T @ r, r.sum(0) 

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
            G = np.asarray(G)
            if G.ndim == 2:
                if np.linalg.norm(G-G.T)/np.linalg.norm(G) > 1e-3:
                    warnings.warn('G should be symmetric, using (G+G.T)/2')
                    G = (G + G.T) / 2
                GX = G @ X_R
                X1_block = self.X.T @ G.sum(0)
            elif G.ndim == 1:
                if not np.all(G >= 0):
                    raise ValueError('weights should be non-negative')
                if not scipy.sparse.issparse(self.X):
                    GX = G[:,None] * X_R
                else:
                    GX = scipy.sparse.diags(G) @ X_R
                X1_block = self.X.T @ G
            else:
                raise ValueError("G should be 1-dim (treated as diagonal) or 2-dim") 
            if scipy.sparse.issparse(GX):
                GX = GX.toarray()
            XX_block = self.X.T @ GX
            G_sum = G.sum()
            
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

    def penalty(self, coef):

        if np.any(coef < self.lower_limits - 1e-3) or np.any(self > self.upper_limits + 1e-3):
            return np.inf
        val = self.lambda_val * (
            self.alpha * (self.penalty_factor * np.fabs(coef)).sum() + 
            (1 - self.alpha) * np.linalg.norm(coef)**2)
        return val
