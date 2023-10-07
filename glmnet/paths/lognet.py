import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import (OneHotEncoder,
                                   LabelEncoder)
from statsmodels.genmod.families import family as sm_family

from .fastnet import FastNetMixin
from ..glm import GLMFamilySpec
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

@dataclass
class LogNet(FastNetMixin):

    modified_newton: bool = False
    _dense = _dense
    _sparse = _sparse

    def __post_init__(self):
        self._family = GLMFamilySpec(base=sm_family.Binomial())

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape): # getcoef.R
        # intercepts will be shape (1,nfits),
        # reshape to (nfits,)
        # specific to binary
        self._fit['a0'] = self._fit['a0'].reshape(-1)
        V = super()._extract_fits(X_shape, response_shape)
        return V

    def _check(self, X, y, check=True):

        X, y, response, offset, weight = super()._check(X, y, check=check)
        encoder = LabelEncoder()
        labels = np.asfortranarray(encoder.fit_transform(response))
        self.classes_ = encoder.classes_
        if len(encoder.classes_) > 2:
            raise ValueError("BinomialGLM expecting a binary classification problem.")
        return X, y, labels, offset, weight

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

        # adjust dim of offset -- seems necessary to get 1d?

        if offset is None:
            offset = response * 0.
        else:
            # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L57
            offset = np.column_stack([offset,-offset])
            
        offset = np.asfortranarray(offset)

        nobs, nvars = design.X.shape

        # add 'kopt' 
        _args['kopt'] = int(self.modified_newton)

        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L42
        nc = 1

        # add 'g'
        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L65
        _args['g'] = offset[:,0]

        # fix intercept and coefs

        _args['a0'] = np.asfortranarray(np.zeros((nc, self.nlambda), float))
        _args['ca'] = np.zeros((nvars*self.nlambda*nc, 1))

        # reshape y
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(_args['y']))
        _args['y'] = y_onehot

        _args['y'] *= sample_weight[:,None]
        # from https://github.com/trevorhastie/glmnet/blob/master/R/lognet.R#L43
        _args['y'] = np.asfortranarray(_args['y'][:,::-1])

        # remove w
        del(_args['w'])
        
        return _args
