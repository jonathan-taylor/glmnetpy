import logging
LOG = False
from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y

from .glmnetpp import wls as dense_wls
from .glmnetpp import spwls as sparse_wls
from .base import (Base,
                   Penalty,
                   Design,
                   _get_design)
from ._utils import _jerr_elnetfit
from .docstrings import (make_docstring,
                         add_dataclass_docstring)
DEBUG = False

@add_dataclass_docstring
@dataclass
class ElNetControl(object):

    thresh: float = 1e-7
    maxit: int = 100000
    big: float = 9.9e35


@dataclass
class ElNetSpec(Penalty):

    fit_intercept: bool = True
    standardize: bool = True
    control: ElNetControl = field(default_factory=ElNetControl)

add_dataclass_docstring(ElNetSpec, subs={'control':'control_elnet'})

@dataclass
class ElNet(BaseEstimator,
            RegressorMixin,
            ElNetSpec):

    def fit(self,
            X,
            y,
            sample_weight=None,
            exclude=[],
            warm=None):

        design = _get_design(X,
                             sample_weight,
                             standardize=self.standardize,
                             intercept=self.fit_intercept)

        if self.lambda_val > 0 or not (np.all(design.centers_ == 0) and np.all(design.scaling_ == 1)):

            self.exclude_ = exclude
            if self.control is None:
                self.control = ElNetControl()
            nobs, nvars = design.X.shape

            if sample_weight is None:
                sample_weight = np.ones(nobs) / nobs

            # because _get_design ignores `standardize` if X is a `Design`, then if `X`
            # is a `Design` this will ignore `self.standardize

            design.X, y = check_X_y(design.X, y,
                                    accept_sparse=['csc'],
                                    multi_output=False,
                                    estimator=self)

            _check_and_set_limits(self, nvars)
            exclude = _check_and_set_vp(self, nvars, exclude)

            if hasattr(self, "_wls_args") and warm is not None:
                args = self._wls_args
                nulldev = self._nulldev
                
                coef, intercept, linear_predictor = warm

                # some sanity checks to see if design matches
                
                if ((design.shape[1] != coef.shape[0] + 1) or
                    (design.shape[0] != linear_predictor.shape[0])): 
                    raise ValueError('dimension mismatch for warm start and design')

                # set the warm start, copying into their arrays will check that sizes are correct

                args['aint'] = intercept
                args['a'][:] = coef.reshape((-1,1))
                args['r'][:] = (sample_weight * (y - linear_predictor)).reshape((-1,1))

            else:
                args, nulldev = _elnet_args(design,
                                            y,
                                            sample_weight,
                                            self.lambda_val,
                                            self.penalty_factor,
                                            alpha=self.alpha,
                                            intercept=self.fit_intercept,
                                            penalty_factor=self.penalty_factor,
                                            exclude=exclude,
                                            lower_limits=self.lower_limits,
                                            upper_limits=self.upper_limits,
                                            thresh=self.control.thresh,
                                            maxit=self.control.maxit)
                self._wls_args = args
                self._nulldev = nulldev
                    
            # override to current values

            args['alpha'] = float(self.alpha)                                 # as.double(alpha)
            args['almc'] = float(self.lambda_val)                             # as.double(lambda)
            args['intr'] = int(self.fit_intercept)                            # as.integer(intercept)
            args['jerr'] = 0                                                  # integer(1) -- mismatch?
            args['maxit'] = int(self.control.maxit)                           # as.integer(maxit)
            args['thr'] = float(self.control.thresh)                          # as.double(thresh)
            args['v'] = np.asarray(sample_weight, float).reshape((-1,1))      # as.double(weights)

            if LOG: logging.debug(f'Elnet warm coef: {args["a"]}, Elnet warm intercept: {args["aint"]}')

            check_resid = True
            if check_resid:
                r1 = args['r'].reshape(-1)
                r2 = sample_weight * (y - design @ np.hstack([args['aint'], args['a'].reshape(-1)])).reshape(-1)
                if np.linalg.norm(r1-r2) / max(np.linalg.norm(r1), 1) > 1e-6:
                    raise ValueError('resid not set correctly')
                
            if scipy.sparse.issparse(design.X):
                wls_fit = sparse_wls(**args)
            else:
                wls_fit = dense_wls(**args)

            # if error code > 0, fatal error occurred: stop immediately
            # if error code < 0, non-fatal error occurred: return error code

            if wls_fit['jerr'] != 0:
                errmsg = _jerr_elnetfit(wls_fit['jerr'], self.control.maxit)
                raise ValueError(errmsg['msg'])

            if LOG: logging.debug(f'Elnet coef: {wls_fit["a"]}, Elnet intercept: {wls_fit["aint"]}')
            beta = scipy.sparse.csc_array(wls_fit['a']) # shape=(1, nvars)
            
            intercept_ = wls_fit['aint']
            result = ElNetResult(a0=wls_fit['aint'],
                                 beta=beta,
                                 df=np.sum(np.abs(beta) > 0),
                                 dim=beta.shape,
                                 lambda_val=self.lambda_val,
                                 dev_ratio=wls_fit['rsqc'],
                                 nulldev=nulldev,
                                 npasses=wls_fit['nlp'],
                                 jerr=wls_fit['jerr'],
                                 nobs=nobs,
                                 sample_weight=sample_weight)
        else:
            # can use LinearRegression

            lm = LinearRegression(fit_intercept=self.fit_intercept)
            if scipy.sparse.issparse(design.X):
                X_s = scipy.sparse.csc_matrix(design.X)
            else:
                X_s = design.X
            lm.fit(X_s, y, sample_weight)
            
            beta = scipy.sparse.csc_array(lm.coef_)

            intercept_ = lm.intercept_
            result = ElNetResult(a0=lm.intercept_,
                                 beta=beta,
                                 df=X.shape[1]+self.fit_intercept,
                                 dim=beta.shape,
                                 lambda_val=0,
                                 dev_ratio=None,
                                 nulldev=np.inf,
                                 npasses=1,
                                 jerr=0,
                                 nobs=X.shape[1],
                                 sample_weight=sample_weight)

        self.result_ = result
        self.design_ = design
        self.coef_ = beta.toarray() / design.scaling_
        self.intercept_ = intercept_ - (self.coef_ * self.design_.centers_).sum()
        self._args = args
        return self

add_dataclass_docstring(ElNet, subs={'control':'control_elnet'})

@add_dataclass_docstring
@dataclass
class ElNetResult(object):

    a0: float 
    beta: scipy.sparse._csc.csc_array 
    df: int
    dim: tuple
    lambda_val: float
    dev_ratio: float
    nulldev: float
    npasses: int
    jerr: int
    nobs: int
    sample_weight: np.ndarray

_elnet_fit_doc = r'''A wrapper around a C++ subroutine which minimizes

.. math::

    1/2 \sum w_i (y_i - X_i^T \beta)^2 + \sum \lambda \gamma_j [(1-\alpha)/2 \beta^2+\alpha|\beta|]

over $\beta$, where $\gamma_j$ is the relative penalty factor on the
j-th variable. If `intercept`, then the term in the first sum is
$w_i (y_i - \beta_0 - X_i^T \beta)^2$, and we are minimizing over both
$\beta_0$ and $\beta$.

None of the inputs are standardized except for `penalty_factor`, which
is standardized so that they sum up to `nvars`.

{0}

Returns
-------

result: ElNetResult

'''.format(make_docstring('X',
                          'y',
                          'sample_weight',
                          'lambda_val',
                          'alpha',
                          'fit_intercept',
                          'penalty_factor',
                          'exclude',
                          'lower_limits',
                          'upper_limits',
                          'thresh',
                          'maxit'))

def _elnet_args(design,
                y,
                sample_weight,
                lambda_val,
                vp, 
                alpha=1.0,
                intercept=True,
                thresh=1e-7,
                maxit=100000,
                penalty_factor=None, 
                exclude=[],
                lower_limits=-np.inf,
                upper_limits=np.inf):
    
    X = design.X
        
    exclude = np.asarray(exclude, np.int32)

    nobs, nvars = X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # compute null deviance
    # sample_weight = sample_weight / sample_weight.sum()
    
    ybar = np.sum(y * sample_weight) / np.sum(sample_weight)
    nulldev = np.sum(sample_weight * (y - ybar)**2)

    # if calling from glmnet.fit(), we do not need to check on exclude
    # and penalty.factor arguments as they have been prepared by glmnet.fit()
    # compute ju
    # assume that there are no constant variables

    ju = np.ones((nvars, 1), np.int32)
    ju[exclude] = 0

    # compute cl from upper and lower limits

    cl = np.asfortranarray([lower_limits,
                            upper_limits], float)

    nx = nvars #  as.integer(nvars)

                                     # From elnet.fit R code
    a  = np.zeros((nvars, 1))        # double(nvars)
    aint = 0.                        # double(1) -- mismatch?
    alm0  = 0.                       # double(1) -- mismatch?
    g = np.zeros((nvars, 1))         # double(nvars) -- mismatch?
    ia = np.zeros((nx, 1), np.int32) # integer(nx)
    iy = np.zeros((nvars, 1), np.int32)   # integer(nvars)     
    iz = 0                           # integer(1) -- mismatch?
    m = 1                            # as.integer(1)
    mm = np.zeros((nvars, 1), np.int32)   # integer(nvars) -- mismatch?
    nino = int(0)                    # integer(1)
    nlp = 0                          # integer(1) -- mismatch?
    r =  (sample_weight * y).reshape((-1,1))
    rsqc = 0.                        # double(1) -- mismatch?
    xv = np.zeros((nvars, 1))        # double(nvars)


    alpha = float(alpha)                            # as.double(alpha)
    almc = float(lambda_val)                        # as.double(lambda)
    intr = int(intercept)                           # as.integer(intercept)
    jerr = 0                                        # integer(1) -- mismatch?
    maxit = int(maxit)                              # as.integer(maxit)
    thr = float(thresh)                             # as.double(thresh)
    v = np.asarray(sample_weight, float).reshape((-1,1))  # as.double(weights)

    a_new = a # .copy() 
    # take out components of x and run C++ subroutine

    _args = {'alm0':alm0,
             'almc':almc,
             'alpha':alpha,
             'm':m,
             'no':nobs,
             'ni':nvars,
             'r':r,
             'xv':xv,
             'v':v,
             'intr':intr,
             'ju':ju,
             'vp':vp,
             'cl':cl,
             'nx':nx,
             'thr':thr,
             'maxit':maxit,
             'a':a,
             'aint':aint,
             'g':g,
             'ia':ia,
             'iy':iy,
             'iz':iz,
             'mm':mm,
             'nino':nino,
             'rsqc':rsqc,
             'nlp':nlp,
             'jerr':jerr}

    _args.update(**_design_wls_args(design))

    return _args, nulldev

def _check_and_set_limits(spec, nvars):

    lower_limits = np.asarray(spec.lower_limits)
    upper_limits = np.asarray(spec.upper_limits)

    lower_limits = np.asarray(lower_limits)
    upper_limits = np.asarray(upper_limits)

    if lower_limits.shape in [(), (1,)]:
        lower_limits = -np.inf * np.ones(nvars)

    if upper_limits.shape in [(), (1,)]:
        upper_limits = np.inf * np.ones(nvars)

    lower_limits = lower_limits[:nvars]
    upper_limits = upper_limits[:nvars]

    if lower_limits.shape[0] < nvars:
        raise ValueError('lower_limits should have shape {0}, but has shape {1}'.format((nvars,),
                                                                                        lower_limits.shape))
    if upper_limits.shape[0] < nvars:
        raise ValueError('upper_limits should have shape {0}, but has shape {1}'.format((nvars,),
                                                                                        upper_limits.shape))
    lower_limits[lower_limits == -np.inf] = -spec.control.big
    upper_limits[upper_limits == np.inf] = spec.control.big

    spec.lower_limits, spec.upper_limits = lower_limits, upper_limits

def _check_and_set_vp(spec, nvars, exclude):

    penalty_factor = spec.penalty_factor

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # check and standardize penalty factors (to sum to nvars)
    _isinf_penalty = np.isinf(penalty_factor)

    if np.any(_isinf_penalty):
        exclude.extend(np.nonzero(_isinf_penalty)[0])
        exclude = np.unique(exclude)

    exclude = list(np.asarray(exclude, int))

    if len(exclude) > 0:
        if max(exclude) >= nvars:
            raise ValueError("Some excluded variables out of range")
        penalty_factor[exclude] = 1 # now can change penalty_factor

    vp = np.maximum(0, penalty_factor).reshape((-1,1))
    vp = (vp * nvars / vp.sum())

    spec.penalty_factor = vp

    return exclude

def _design_wls_args(design):
    if not scipy.sparse.issparse(design.X):
        return {'x':design.X}
    else:
        return {'x_data_array':design.X.data,
                'x_indices_array':design.X.indices,
                'x_indptr_array':design.X.indptr,
                'xm':design.centers_,
                'xs':design.scaling_}
