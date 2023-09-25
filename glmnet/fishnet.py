import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y

from .gaussnet import FastNetMixin
from .docstrings import (make_docstring,
                         add_dataclass_docstring)

from ._fishnet import fishnet as fishnet_dense
from ._fishnet import spfishnet as fishnet_sparse

@dataclass
class FishNet(FastNetMixin):

    univariate_beta: bool = True
    type_logistic: Literal['Newton', 'modified_Newton'] = 'Newton'
    _dense = fishnet_dense
    _sparse = fishnet_sparse

    # private methods

    def _check(self, X, y):

        if np.any(y < 0):
            raise ValueError("negative responses encountered;  not permitted for Poisson family")
        return super()._check(X, y)

    def _wrapper_args(self,
                      design,
                      y,
                      sample_weight,
                      offset,
                      exclude=[]):

        _args = super()._wrapper_args(design,
                                      y,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # adjust dim of offset -- seems necessary to get 1d?

        if offset is None:
            offset = y * 0.
            if self.univariate_beta:
                offset = offset[:,:1]

        if self.univariate_beta:
            if offset.ndim == 2 and offset.shape[1] != 1:
                raise ValueError('for binary classification as univariate, offset should be 1d')
        offset = np.asfortranarray(offset)

        nobs, nvars = design.X.shape

        # add 'kopt' 
        _args['kopt'] = {'Newton':0,
                         'modified_Newton':1}[self.type_logistic]

        # add 'g'
        _args['g'] = offset
        
        # fix intercept and coefs

        if self.univariate_beta:
            nc = 1
        _args['a0'] = np.asfortranarray(np.zeros((nc, self.nlambda), float))
        _args['ca'] = np.zeros((nvars*self.nlambda*nc, 1))

        # reshape y
        _args['y'] = np.asfortranarray(_args['y'].reshape((nobs, len(self.categories_))))
#        probably should scale these?
#        _args['y'] *= sample_weight[:,None]

        # remove w
        del(_args['w'])
        
        return _args

    def _wrapper_args(self,
                      design,
                      y,
                      sample_weight,
                      offset,
                      exclude=[]):
        
        if offset is None:
            offset = 0. * y
        offset = np.asfortranarray(offset.reshape((-1,1)))

        return super()._wrapper_args(design,
                                     y,
                                     sample_weight,
                                     offset,
                                     exclude=exclude)

