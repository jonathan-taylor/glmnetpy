import logging
import warnings

from typing import Literal
from dataclasses import dataclass, field
   
import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.utils import check_X_y

from ._gaussnet import gaussnet as gaussnet_dense
from ._gaussnet import spgaussnet as gaussnet_sparse

from .base import _get_design
from .glmnet import GLMNet
from .elnet import (_check_and_set_limits,
                    _check_and_set_vp,
                    _design_wrapper_args)

from ._utils import _jerr_elnetfit
from .docstrings import (make_docstring,
                         add_dataclass_docstring)

@dataclass
class FastNetMixin(GLMNet): # base class for C++ path methods

    def fit(self,
            X,
            y,
            sample_weight=None,
            offset=None,
            exclude=[]):

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
            self.control = GLMNetControl()

        nobs, nvars = design.X.shape

        if sample_weight is None:
            sample_weight = np.ones(nobs) / nobs

        _check_and_set_limits(self, nvars)
        exclude = _check_and_set_vp(self, nvars, exclude)

        self._args = self._wrapper_args(design,
                                        y,
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

        self._args.update(**_design_wrapper_args(design))

        if scipy.sparse.issparse(design.X):
            self._fit = self._sparse(**self._args)
        else:
            self._fit = self._dense(**self._args)

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if self._fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(self._fit['jerr'], self.control.maxit)
            if self.control.logging: logging.debug(errmsg['msg'])

        # extract the coefficients
        
        result = self._extract_fits()
        nvars = design.X.shape[1]
        self.coefs_ = result['coefs'] / design.scaling_[None,:]
        self.intercepts_ = (result['intercepts'] -
                            (self.coefs_ * design.centers_[None,:]).sum(1))

        self.lambda_values_ = result['lambda_values']
        nfits = self.lambda_values_.shape[0]
        dev_ratios_ = self._fit['dev'][:nfits]
        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = result['df']
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        return self

    # private methods

    def _extract_fits(self): # getcoef.R
        _fit, _args = self._fit, self._args
        nx = _args['nx']
        nvars = _args['ni']
        nfits = _fit['lmu']

        if nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        nin = _fit['nin'][:nfits]
        ninmax = max(nin)
        lambda_values = _fit['alm'][:nfits]

        if ninmax > 0:
            if _fit['ca'].ndim == 1: # logistic is like this
                coefs = _fit['ca'][:(nx*nfits)].reshape(nfits, nx)
            else:
                coefs = _fit['ca'][:,:nfits].T
            df = (np.fabs(coefs) > 0).sum(1)
            # this is order variables appear in the path
            # reorder to set original coords
            active_seq = _fit['ia'][:nfits]
            _argsort = np.argsort(active_seq)
            coefs = coefs[:, _argsort]
            intercepts = _fit['a0'][:nfits]

        return {'coefs':coefs,
                'intercepts':intercepts,
                'df':df,
                'lambda_values':lambda_values}

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
        raise NotImplementedError('abstract method')


@dataclass
class GaussNet(FastNetMixin):

    type_gaussian: Literal['covariance', 'naive'] = None

    _dense = gaussnet_dense
    _sparse = gaussnet_sparse

    # private methods

    def _extract_fits(self):
        self._fit['dev'] = self._fit['rsq'] # gaussian fit calls it rsq
        return super()._extract_fits()
        
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
            is_offset = False
        else:
            offset = np.asarray(offset).astype(float)
            y = y - offset # makes a copy, does not modify y
        y = y.copy().reshape((-1,1))
        offset = np.asfortranarray(offset)

        # compute jd
        # assume that there are no constant variables

        jd = np.ones((nvars, 1), np.int32)
        jd[exclude] = 0
        jd = np.nonzero(jd)[0].astype(np.int32)

        # compute cl from upper and lower limits
        cl = np.asarray([lower_limits,
                         upper_limits], float)

        if self.type_gaussian is None:
            if nvars < 500:
                self.type_gaussian = 'covariance'
            else:
                self.type_gaussian = 'naive'

        ka = {'covariance':1,
                'naive':2}[self.type_gaussian]

        # compute nulldeviance

        ybar = (y * sample_weight).sum() / sample_weight.sum()
        nulldev = ((y - ybar)**2 * sample_weight).sum() / sample_weight.sum()

        if nulldev == 0:
            raise ValueError("y is constant; gaussian glmnet fails at standardization step")

        # all but the X -- this is set below

        nx = min((nvars+1)*2+20, nvars)

        _args = {'ka':ka,
                 'parm':float(alpha),
                 'ni':nvars,
                 'no':nobs,
                 'y':y,
                 'w':sample_weight.reshape((-1,1)),
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
                 'rsq':np.zeros((nlambda, 1)),
                 'alm':np.zeros((nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        _args.update(**_design_wrapper_args(design))

        return _args

