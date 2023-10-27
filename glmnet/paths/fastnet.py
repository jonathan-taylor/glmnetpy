import logging
import warnings

from typing import Literal
from dataclasses import dataclass, field, asdict
   
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from ..base import _get_design
from ..glmnet import GLMNet
from ..glm import GLMState
from ..elnet import (_check_and_set_limits,
                    _check_and_set_vp,
                    _design_wrapper_args)

from .._utils import _jerr_elnetfit
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

@dataclass
class FastNetControl(object):

    fdev: float = 1e-5
    eps: float = 1e-6
    big: float = 9.9e35
    mnlam: int = 5
    devmax: float = 0.999
    pmin: float = 1e-9
    exmx: float = 250.
    itrace: int = 0
    prec: float = 1e-10
    mxit: int = 100
    epsnr: float = 1e-6
    mxitnr: int = 25
    # thresh & logging not part of glmnet.control but used in the wrapper
    maxit: int = 100000
    thresh: float = 1e-7
    logging: bool = False
    
@dataclass
class FastNetMixin(GLMNet): # base class for C++ path methods

    lambda_min_ratio: float = None
    nlambda: int = 100
    df_max: int = None
    control: FastNetControl = field(default_factory=FastNetControl)

    def fit(self,
            X,
            y,
            sample_weight=None, # ignored
            interpolation_grid=None):
    
        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        X, y, response, offset, weight = self._check(X, y)

        if not scipy.sparse.issparse(X):
            X = np.asfortranarray(X)
        
        # the C++ path codes handle standardization
        # themselves so we shouldn't handle it at this level
        if not hasattr(self, "design_"):
            if weight is None:
                weight = np.ones(X.shape[0])
            self.design_ = design = _get_design(X,
                                                weight,
                                                standardize=False,
                                                intercept=False)
        else:
            design = self.design_


        if self.df_max is None:
            self.df_max = X.shape[1] + 1
            
        if self.control is None:
            self.control = FastNetControl()

        nobs, nvars = design.X.shape

        sample_weight = weight
        
        _check_and_set_limits(self, nvars)
        self.exclude = _check_and_set_vp(self, nvars, self.exclude)

        self.pb = tqdm(total=self.nlambda)
        self._args = self._wrapper_args(design,
                                        response,
                                        sample_weight,
                                        offset=offset,
                                        exclude=self.exclude)

        self._args.update(**_design_wrapper_args(design))

        # set control args
        D = asdict(self.control)
        del(D['maxit']) # maxit is not in glmnet.control
        del(D['thresh']) # thresh is not in glmnet.control
        del(D['logging']) # logging is not in glmnet.control
        self._args.update(**D)

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
        
        result = self._extract_fits(X.shape, response.shape)
        nvars = design.X.shape[1]

        self.coefs_ = result['coefs']
        self.intercepts_ = result['intercepts']
            
        self.state_ = GLMState(self.coefs_[-1],
                               self.intercepts_[-1])

        self.lambda_values_ = result['lambda_values']
        nfits = self.lambda_values_.shape[0]
        dev_ratios_ = self._fit['dev'][:nfits]
        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = result['df']
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        # set lambda_max

        # following https://github.com/trevorhastie/glmnet/blob/master/R/glmnet.R#L523
        # will work with equispaced values on log scale

        if len(self.lambda_values_) > 2 and self.lambda_values is None:
            self.lambda_values_[0] = self.lambda_values_[1]**2 / self.lambda_values_[2]

        self.lambda_max_ = self.lambda_values_[0]

        if interpolation_grid is not None:
            self.coefs_, self.intercepts_ = self.interpolate_coefs(interpolation_grid)

        return self

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape): # getcoef.R
        _fit, _args = self._fit, self._args
        nvars = X_shape[1]
        nfits = _fit['lmu']

        if nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        nin = _fit['nin'][:nfits]
        ninmax = max(nin)
        lambda_values = _fit['alm'][:nfits]

        if ninmax > 0:
            if _fit['ca'].ndim == 1: # logistic is like this
                unsort_coefs = _fit['ca'][:(nvars*nfits)].reshape(nfits, nvars)
            else:
                unsort_coefs = _fit['ca'][:,:nfits].T
            df = (np.fabs(unsort_coefs) > 0).sum(1)

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1

            coefs = np.zeros((nfits, nvars))
            coefs[:, active_seq] = unsort_coefs[:, :len(active_seq)]
            intercepts = _fit['a0'][:nfits]

        return {'coefs':coefs,
                'intercepts':intercepts,
                'df':df,
                'lambda_values':lambda_values}
 
    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset, # ignored, but subclasses use it
                      exclude=[]):

        sample_weight = np.asfortranarray(sample_weight)
        
        X = design.X
        nobs, nvars = X.shape

        if self.lambda_min_ratio is None:
            if nobs < nvars:
                self.lambda_min_ratio = 1e-2
            else:
                self.lambda_min_ratio = 1e-4

        if self.lambda_values is None:
            if self.lambda_min_ratio > 1:
                raise ValueError('lambda_min_ratio should be less than 1')
            flmin = float(self.lambda_min_ratio)
            ulam = np.zeros((1, 1))
        else:
            flmin = 1.
            if np.any(self.lambda_values < 0):
                raise ValueError('lambdas should be non-negative')
            ulam = np.asfortranarray(np.sort(self.lambda_values)[::-1].reshape((-1, 1)))
            self.nlambda = self.lambda_values.shape[0]

        if response.ndim == 1:
            response = response.reshape((-1,1))

        # compute jd
        # assume that there are no constant variables

        if len(exclude) > 0:
            jd = np.hstack([len(exclude), exclude]).astype(np.int32)
        else:
            jd = np.array([0], np.int32)
            
        # compute cl from upper and lower limits

        if not np.all(self.lower_limits <= 0):
            raise ValueError('lower limits should be <= 0')

        if not np.all(self.upper_limits >= 0):
            raise ValueError('upper limits should be >= 0')

        cl = np.asarray([self.lower_limits,
                         self.upper_limits], float)

        if np.any(cl[0] == 0) or np.any(cl[-1] == 0):
            self.control.fdev = 0

        # all but the X -- this is set below

        # isn't this always nvars?
        # should have a df_max arg
        nx = min(self.df_max*2+20, nvars)

        _args = {'parm':float(self.alpha),
                 'ni':nvars,
                 'no':nobs,
                 'y':response,
                 'w':sample_weight.reshape((-1,1)),
                 'jd':jd,
                 'vp':self.penalty_factor.reshape((-1,1)),
                 'cl':np.asfortranarray(cl),
                 'ne':self.df_max,
                 'nx':nx,
                 'nlam':self.nlambda,
                 'flmin':flmin,
                 'ulam':ulam,
                 'thr':float(self.control.thresh),
                 'isd':int(self.standardize),
                 'intr':int(self.fit_intercept),
                 'maxit':int(self.control.maxit),
                 'pb':self.pb,
                 'lmu':0, # these asfortran calls not necessary -- nullop
                 'a0':np.asfortranarray(np.zeros((self.nlambda, 1), float)),
                 'ca':np.asfortranarray(np.zeros((nx, self.nlambda))),
                 'ia':np.zeros((nx, 1), np.int32),
                 'nin':np.zeros((self.nlambda, 1), np.int32),
                 'nulldev':0.,
                 'dev':np.zeros((self.nlambda, 1)),
                 'alm':np.zeros((self.nlambda, 1)),
                 'nlp':0,
                 'jerr':0,
                 }

        return _args


@dataclass
class MultiFastNetMixin(FastNetMixin): # paths with multiple responses

    def predict(self,
                X,
                prediction_type='link' # ignored except checking valid
                ):

        if prediction_type not in ['response', 'link']:
            raise ValueError("prediction should be one of 'response' or 'link'")
        
        term1 = np.einsum('ijk,lj->ilk',
                          self.coefs_,
                          X)
        fits = term1 + self.intercepts_[:, None, :]
        fits = np.transpose(fits, [1,0,2])

        # make return based on original
        # promised number of lambdas
        # pad with last value

        # if possible we might want to do less than `self.nlambda`
        
        if self.lambda_values is not None:
            nlambda = self.lambda_values.shape[0]
        else:
            nlambda = self.nlambda

        value = np.empty((fits.shape[0],
                          nlambda,
                          fits.shape[2]), float) * np.nan
        value[:,:fits.shape[1]] = fits
        value[:,fits.shape[1]:] = fits[:,-1][:,None]

        return value

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        _fit, _args = self._fit, self._args
        nvars = X_shape[1]
        nresp = response_shape[1]
        nfits = _fit['lmu']
        if nfits < 1:
            warnings.warn("an empty model has been returned; probably a convergence issue")

        nin = _fit['nin'][:nfits]
        ninmax = max(nin)
        lambda_values = _fit['alm'][:nfits]

        if ninmax > 0:
            unsort_coefs = _fit['ca'][:(nresp*nvars*nfits)].reshape(nfits,
                                                                    nresp,
                                                                    nvars)
            unsort_coefs = np.transpose(unsort_coefs, [0,2,1])
            df = ((unsort_coefs**2).sum(2) > 0).sum(1)

            # this is order variables appear in the path
            # reorder to set original coords

            active_seq = _fit['ia'].reshape(-1)[:ninmax] - 1

            coefs = np.zeros((nfits, nvars, nresp))
            coefs[:, active_seq] = unsort_coefs[:, :len(active_seq)]
            intercepts = _fit['a0'][:,:nfits].T

        return {'coefs':coefs,
                'intercepts':intercepts,
                'df':df,
                'lambda_values':lambda_values}


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

        # ensure shapes are correct

        (nobs, nvars), nr = design.X.shape, response.shape[1]
        _args['a0'] = np.asfortranarray(np.zeros((nr, self.nlambda), float))
        _args['ca'] = np.zeros((self.nlambda * nr * nvars, 1))
        _args['y'] = np.asfortranarray(_args['y'].reshape((nobs, nr)))

        return _args
