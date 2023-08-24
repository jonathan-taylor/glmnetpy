import warnings
from typing import Union, Optional
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse

from .docstrings import add_dataclass_docstring

@add_dataclass_docstring
@dataclass
class Design(object):

    """
    Linear map representing multiply with [1,X] and its transpose.
    """

    X: Union[np.ndarray, scipy.sparse._csc.csc_array]
    weights: np.ndarray
    standardize: bool = False
    
    def __post_init__(self):

        n, p = self.shape = self.X.shape
        X, weights = self.X, self.weights

        # if standardizing, then the effective matrix is
        # (X - (1'W)^{-1} 1 W'X) S^{-1}
        # (X - 1 xm') @ diag(1/xs) = XS^{-1} - 1 xm/xs'

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
        # A = XS^{-1} - 1 (xm/xs)'
        # GA = GXS^{-1} - G.sum(0) (xm/xs)'
        # A'G1 = S^{-1}X'G1 - 1'G1 (xm/xs)'
        # A'GA = S^{-1}X'GXS^{-1} - S^{-1}X'G1 xm/xs' - xm/xs 1'GXS^{-1} + xm/xs 1'G1 (xm/xs)'
        
        n, p = self.X.shape
        
        if columns is None:
            X_E = self.X
        else:
            X_E = self.X[:,columns]
        if G is None:
            
            XX_block = self.X.T @ X_E # have to assume this is not too expensive
            X1_block = self.X.sum(0) # X'1
            G_sum = n
            
        else:
            G = np.asarray(G)
            if G.ndim == 2:
                if np.linalg.norm(G-G.T)/np.linalg.norm(G) > 1e-3:
                    warnings.warn('G should be symmetric, using (G+G.T)/2')
                    G = (G + G.T) / 2
                GX = G @ self.X
            elif G.ndim == 1:
                if not np.all(G >= 0):
                    raise ValueError('weights should be non-negative')
                if not scipy.sparse.issparse(self.X):
                    GX = G[:,None] * self.X 
                else:
                    GX = scipy.sparse.diags(G) @ self.X  
            else:
                raise ValueError("G should be 1-dim (treated as diagonal) or 2-dim") 
            if scipy.sparse.issparse(GX):
                GX = GX.toarray()
            XX_block, X1_block = self.adjoint_map(GX) # assuming that this not too expensive, same "cost" as without weights
            G_sum = G.sum()
            
        if scipy.sparse.issparse(XX_block):
            XX_block = XX_block.toarray()
            
        # correct XX_block for standardize
        
        XX_block -= (np.multiply.outer(X1_block, self.xm) + np.multiply.outer(self.xm, X1_block))
        XX_block += np.multiply.outer(self.xm, self.xm) * G_sum
        X1_block -= G_sum * self.xm / self.xs
        
        Q = np.zeros((XX_block.shape[0] + 1,)*2)
        Q[1:,1:] = XX_block
        Q[0,1:] = Q[1:,0] = X1_block
        Q[0,0] = G_sum
        
        return Q
    # private methods
    
    def _wls_args(self):
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
    
