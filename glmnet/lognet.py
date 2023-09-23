import logging

from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (LabelEncoder,
                                   OneHotEncoder)
from sklearn.utils import check_X_y

from ._lognet import lognet as lognet_dense
from ._lognet import splognet as lognet_sparse

from .base import (Penalty,
                   Design,
                   _get_design)
from .glmnet import GLMNet
from .elnet import (_check_and_set_limits,
                    _check_and_set_vp,
                    _design_wrapper_args)

from ._utils import _jerr_elnetfit
from .docstrings import (make_docstring,
                         add_dataclass_docstring)

@dataclass
class LogNet(GLMNet):

    def fit(self,
            X,
            y,
            sample_weight=None,
            offset=None,
            exclude=[],
            warm=None,
            check=True):

        if not hasattr(self, "design_"):
            self.design_ = design = _get_design(X,
                                                sample_weight,
                                                standardize=self.standardize,
                                                intercept=self.fit_intercept)
        else:
            design = self.design_

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        X, y = self._check(X, y)

        self.exclude_ = exclude
        if self.control is None:
            self.control = ElNetControl()

        nobs, nvars = design.X.shape

        if sample_weight is None:
            sample_weight = np.ones(nobs) / nobs

        # because _get_design ignores `standardize` if X is a `Design`, then if `X`
        # is a `Design` this will ignore `self.standardize

        if check:
            design.X, y = check_X_y(design.X, y,
                                    accept_sparse=['csc'],
                                    multi_output=False,
                                    estimator=self)

        _check_and_set_limits(self, nvars)
        exclude = _check_and_set_vp(self, nvars, exclude)

        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(y.reshape((-1,1))))
        self.categories_ = encoder.categories_[0]

        self._args = _lognet_wrapper_args(design,
                                          y_onehot,
                                          sample_weight,
                                          self.lambda_values,
                                          alpha=self.alpha,
                                          offset=offset,
                                          fit_intercept=self.fit_intercept,
                                          penalty_factor=self.penalty_factor,
                                          exclude=exclude,
                                          lower_limits=self.lower_limits,
                                          upper_limits=self.upper_limits,
                                          thresh=self.control.thresh,
                                          maxit=self.control.maxit)

        if scipy.sparse.issparse(design.X):
            self._fit = lognet_sparse(**self._args)
        else:
            self._fit = lognet_dense(**self._args)

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if self._fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(self._fit['jerr'], self.control.maxit)
            if self.control.logging: logging.debug(errmsg['msg'])

        _nfits = self._fit['lmu']
        if _nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        # extract the coefficients
        
        nvars = design.X.shape[1]
        ncat = self.categories_.shape[0]
        coefs_ = np.ascontiguousarray(self._fit['ca'])
        self.coefs_ = coefs_[:(nvars*_nfits*ncat)].reshape((_nfits, nvars, ncat))[:,:,1]
        self.lambda_values_ = self._fit['alm'][:_nfits]
        self.lambda_values_[0] = self.lambda_values_[1] # lambda_max not set
        dev_ratios_ = self._fit['dev'][:_nfits]
        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = (self.coefs_ != 0).sum(1)
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        return self

def _lognet_wrapper_args(design,
                         y_onehot,
                         sample_weight,
                         lambda_values,
                         nlambda=100,
                         alpha=1.0,
                         offset=None,
                         lambda_min_ratio=None,
                         fit_intercept=True,
                         standardize=True,
                         thresh=1e-7,
                         maxit=100000,
                         penalty_factor=None, 
                         exclude=[],
                         lower_limits=-np.inf,
                         type_logistic='Newton',
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
        offset = y_onehot * 0.
    offset = np.asfortranarray(offset)
    
    # compute jd
    # assume that there are no constant variables

    jd = np.ones((nvars, 1), np.int32)
    jd[exclude] = 0

    # compute cl from upper and lower limits

    # remember to switch C++ for cl for other families
    cl = np.asarray([lower_limits,
                     upper_limits], float)

    kopt = {'Newton':0,
            'modified_Newton':1}[type_logistic]

    nc = y_onehot.shape[1]

    # take out components of x and run C++ subroutine

    # from https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/lognet.R#L80
    # and https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/src/elnet_exp.cpp#L124

    # all but the X -- this is set below

    nx = min((nvars+1)*2+20, nvars)

    _args = {'parm':float(alpha),
             'ni':nvars,
             'no':nobs,
             'y':y_onehot,
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

    _args.update(**_design_wrapper_args(design))

    return _args

