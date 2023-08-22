from typing import Union
from dataclasses import dataclass

import numpy as np
import scipy.sparse

@dataclass
class DesignSpec(object):

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

    def get_eta(self,
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
