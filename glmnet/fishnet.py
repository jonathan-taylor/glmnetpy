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
            offset = 0. * y
        offset = np.asfortranarray(offset.reshape((-1,1)))

        # compute jd
        # assume that there are no constant variables

        jd = np.ones((nvars, 1), np.int32)
        jd[exclude] = 0
        jd = np.nonzero(jd)[0].astype(np.int32)

        cl = np.asarray([lower_limits,
                         upper_limits], float)

        # all but the X -- this is set below

        nx = min((nvars+1)*2+20, nvars)

        _args = {'parm':float(alpha),
                 'ni':nvars,
                 'no':nobs,
                 'y':(y * 1.).reshape((-1,1)),
                 'w':sample_weight.reshape((-1,1)),
                 'g':offset,
                 'jd':jd,
                 'vp':penalty_factor,
                 'cl':np.asfortranarray(cl),
                 'ne':nvars+1,
                 'nx':nx,
                 'nlam':nlambda,
                 'flmin':flmin,
                 'ulam':ulam,
                 'thr':float(thresh),
                 'isd':int(standardize),
                 'intr':int(fit_intercept),
                 'maxit':int(maxit),
                 'pb':None,
                 'lmu':0, # these asfortran calls not necessary -- nullop
                 'a0':np.asfortranarray(np.zeros((nlambda, 1), float)),
                 'ca':np.asfortranarray(np.zeros((nx, nlambda))),
                 'ia':np.zeros((nx, 1), np.int32),
                 'nin':np.zeros((nlambda, 1), np.int32),
                 'nulldev':float(0),
                 'dev':np.zeros((nlambda, 1)),
                 'alm':np.zeros((nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        return _args

