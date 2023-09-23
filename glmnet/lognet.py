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

from ._lognet import lognet as lognet_dense
from ._lognet import splognet as lognet_sparse

@dataclass
class LogNet(FastNetMixin):

    univariate_beta: bool = True
    type_logistic: Literal['Newton', 'modified_Newton'] = 'Newton'
    _dense = lognet_dense
    _sparse = lognet_sparse

    # private methods

    def _extract_fits(self): # getcoef.R
        # intercepts will be shape (1,nfits),
        # reshape to (nfits,)
        # specific to binary
        self._fit['a0'] = self._fit['a0'].reshape(-1)
        return super()._extract_fits()

    def _check(self, X, y):

        X, y = super()._check(X, y)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(y.reshape((-1,1))))
        self.categories_ = encoder.categories_[0]
        if self.categories_.shape[0] > 2:
            raise ValueError('use multnet for multinomial')
        return X, y_onehot

    def _wrapper_args(self,
                      design,
                      y,
                      sample_weight,
                      lambda_values,
                      offset,
                      nlambda=100,
                      alpha=1.0,
                      lambda_min_ratio=None,
                      fit_intercept=True,
                      standardize=True,
                      thresh=1e-7,
                      maxit=100000,
                      penalty_factor=None, 
                      exclude=[],
                      lower_limits=-np.inf,
                      upper_limits=np.inf):
        
        X = design.X

        nobs, nvars = X.shape

        if lambda_min_ratio is None:
            if nobs < nvars:
                lambda_min_ratio = 1e-2
            else:
                lambda_min_ratio = 1e-4

        if lambda_values is None:
            if lambda_min_ratio > 1:
                raise ValueError('lambda_min_ratio should be less than 1')
            flmin = float(lambda_min_ratio)
            ulam = np.zeros((1, 1))
        else:
            flmin = 1.
            if np.any(lambda_values < 0):
                raise ValueError('lambdas should be non-negative')
            ulam = np.sort(lambda_values)[::-1].reshape((-1, 1))

        if penalty_factor is None:
            penalty_factor = np.ones(nvars)

        if offset is None:
            offset = y * 0.
            if self.univariate_beta:
                offset = offset[:,:1]

        if self.univariate_beta:
            if offset.ndim == 2 and offset.shape[1] != 1:
                raise ValueError('for binary classification as univariate, offset should be 1d')
        offset = np.asfortranarray(offset)

        # compute jd
        # assume that there are no constant variables

        jd = np.ones((nvars, 1), np.int32)
        jd[exclude] = 0
        jd = np.nonzero(jd)[0].astype(np.int32)

        cl = np.asarray([lower_limits,
                         upper_limits], float)

        kopt = {'Newton':0,
                'modified_Newton':1}[self.type_logistic]

        nc = y.shape[1]
        if self.univariate_beta:
            nc = 1

        # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L80
        # and https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/src/elnet_exp.cpp#L124

        # all but the X -- this is set below

        nx = min((nvars+1)*2+20, nvars)

        _args = {'parm':float(alpha),
                 'ni':nvars,
                 'no':nobs,
                 'y':y,
                 'g':offset,
                 'jd':jd,
                 'vp':penalty_factor,
                 'cl':cl,
                 'ne':nvars+1,
                 'nx':nx,
                 'nlam':nlambda,
                 'flmin':flmin,
                 'ulam':ulam,
                 'thr':float(thresh),
                 'isd':int(standardize),
                 'intr':int(fit_intercept),
                 'maxit':int(maxit),
                 'kopt':kopt,
                 'pb':None,
                 'lmu':0,
                 'a0':np.asfortranarray(np.zeros((nc, nlambda), float)),
                 'ca':np.zeros((nx*nlambda*nc, 1)),
                 'ia':np.zeros((nx, 1), np.int32),
                 'nin':np.zeros((nlambda, 1), np.int32),
                 'nulldev':0.,
                 'dev':np.zeros((nlambda, 1)),
                 'alm':np.zeros((nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        return _args

