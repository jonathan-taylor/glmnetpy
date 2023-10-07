import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y

from .fastnet import MultiFastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

@dataclass
class MultiClassNet(MultiFastNetMixin):

    standardize_response: bool = False
    grouped: bool = False
    univariate_beta: bool = True
    type_logistic: Literal['Newton', 'modified_Newton'] = 'Newton'
    _dense = _dense
    _sparse = _sparse

    # private methods

    def _check(self, X, y):

        X, y, response, offset, weight = super()._check(X, y)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(y.reshape((-1,1))))
        self.categories_ = encoder.categories_[0]
        return X, y, y_onehot, offset, weight

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        # center the intercepts -- any constant
        # added does not affect class probabilities
        
        self._fit['a0'] = self._fit['a0'] - self._fit['a0'].mean(0)[None,:]
        return super()._extract_fits(X_shape, response_shape)

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        if offset is None:
            offset = response * 0.
        if offset.shape != response.shape:
            raise ValueError('offset shape should match one-hot response shape')
        offset = np.asfortranarray(offset)

        # add 'kopt' 
        _args['kopt'] = {'Newton':0,
                         'modified_Newton':1}[self.type_logistic]
        # if grouped, we set kopt to 2
        if self.grouped:
            _args['kopt'] = 2

        # add 'g'
        _args['g'] = offset

        # take care of weights
        _args['y'] *= sample_weight[:,None]

        # remove w
        del(_args['w'])

        return _args

