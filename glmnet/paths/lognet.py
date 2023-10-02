import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from .fastnet import FastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

@dataclass
class LogNet(FastNetMixin):

    modified_newton: bool = False
    _dense = _dense
    _sparse = _sparse

    univariate_beta = True # not tested as False

    # private methods

    def _extract_fits(self,
                      X_shape,
                      y_shape): # getcoef.R
        # intercepts will be shape (1,nfits),
        # reshape to (nfits,)
        # specific to binary
        self._fit['a0'] = self._fit['a0'].reshape(-1)
        V = super()._extract_fits(X_shape, y_shape)
        return V

    def _check(self, X, y):

        X, y = super()._check(X, y)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(y.reshape((-1,1))))
        self.categories_ = encoder.categories_[0]
        if self.categories_.shape[0] > 2:
            raise ValueError('use MultiClassNet for multinomial')
        return X, y_onehot

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
        _args['kopt'] = int(self.modified_newton)

        # add 'g'
        _args['g'] = offset
        
        # fix intercept and coefs

        if self.univariate_beta:
            nc = 1
        else:
            nc = len(self.categories_)
        _args['a0'] = np.asfortranarray(np.zeros((nc, self.nlambda), float))
        _args['ca'] = np.zeros((nvars*self.nlambda*nc, 1))

        # reshape y
        _args['y'] = np.asfortranarray(_args['y'].reshape((nobs, len(self.categories_))))

        # probably should scale these?

        _args['y'] *= sample_weight[:,None]

        # remove w
        del(_args['w'])
        
        return _args
