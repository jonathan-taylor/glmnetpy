from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin

from .glmnetpp import wls as dense_wls
from .glmnetpp import spwls as sparse_wls
from .base import Base, Penalty, Options, Design
from ._utils import _jerr_elnetfit, _dataclass_from_parent
from .docstrings import make_docstring, add_dataclass_docstring


@add_dataclass_docstring
@dataclass
class ElNetControl(object):

    thresh: float = 1e-7
    maxit: int = 100000
    big: float = 9.9e35


@dataclass
class ElNetSpec(Options,
                Penalty):

    control: ElNetControl = field(default_factory=ElNetControl)

add_dataclass_docstring(ElNetSpec, subs={'control':'control_elnet'})

@dataclass
class ElNetEstimator(BaseEstimator,
                     RegressorMixin,
                     ElNetSpec):

    def fit(self, X, y, weights=None, warm=None):

        if self.control is None:
            self.control = ElNetControl()
        elif type(self.control) == dict:
            self.control = _dataclass_from_parent(ElNetControl,
                                                  self.control)
        if self.exclude is None:
            self.exclude = []
            
        nobs, nvars = X.shape
        
        if weights is None:
            weights = np.ones(nobs)

        # because _get_design ignores `standardize` if X is a `Design`, then if `X`
        # is a `Design` this will ignore `self.standardize
        design = _get_design(X, weights, standardize=self.standardize)

        _check_and_set_limits(self, nvars)
        _check_and_set_vp(self, nvars)
        
        args, nulldev = _wls_args(self, design, y, weights, warm)

        if scipy.sparse.issparse(design.X):
            wls_fit = sparse_wls(**args)
        else:
            wls_fit = dense_wls(**args)

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if wls_fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(wls_fit['jerr'], self.control.maxit)
            raise ValueError(errmsg['msg'])

        warm_args = {}
        for key in ["almc", "r", "xv", "ju", "vp",
                    "cl", "nx", "a", "aint", "g",
                    "ia", "iy", "iz", "mm", "nino",
                    "rsqc", "nlp"]:
                warm_args[key] = wls_fit[key]

        warm_args['m'] = args['m'] # isn't this always 1?
        warm_args['no'] = nobs
        warm_args['ni'] = nvars

        warm_fit = ElNetWarmStart(**warm_args)

        beta = scipy.sparse.csc_array(wls_fit['a']) # shape=(1, nvars)

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
                             warm_fit=warm_fit)

        self.result_ = result
        return self

    def _wls_args(self, design, y, weights, warm=None):
        return _wls_args(self, design, y, weights, warm)

add_dataclass_docstring(ElNetEstimator, subs={'control':'control_elnet'})

@dataclass
class ElNetWarmStart(object):

    almc: float
    r: np.ndarray
    xv: np.ndarray
    ju: np.ndarray
    vp: np.ndarray
    cl: np.ndarray
    nx: int
    a: np.ndarray
    aint: float
    g: np.ndarray
    ia: np.ndarray
    iy: np.ndarray
    iz: int
    mm: np.ndarray
    nino: int
    rsqc: float
    nlp: int
    m: int
    no: int
    ni: int

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
    warm_fit: dict

def elnet_fit(X,
              y,
              weights,
              lambda_val,
              alpha=1.0,
              intercept=True,
              penalty_factor=None, 
              exclude=[],
              lower_limits=-np.inf,
              upper_limits=np.inf,
              thresh=1e-7,
              maxit=100000,
              warm=None):

    control = ElNetControl(thresh=thresh,
                           maxit=maxit)
    
    problem = ElNetSpec(X=X,
                        y=y,
                        weights=weights,
                        lambda_val=lambda_val,
                        alpha=alpha,
                        intercept=intercept,
                        penalty_factor=penalty_factor,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        exclude=exclude,
                        control=control)
    
    return problem.fit()

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
                          'weights',
                          'lambda_val',
                          'alpha',
                          'intercept',
                          'penalty_factor',
                          'exclude',
                          'lower_limits',
                          'upper_limits',
                          'thresh',
                          'maxit',
                          'warm'))

elnet_fit.__doc__ = _elnet_fit_doc

def _elnet_args(design,
                y,
                weights,
                lambda_val,
                vp, 
                alpha=1.0,
                intercept=True,
                thresh=1e-7,
                maxit=100000,
                penalty_factor=None, 
                exclude=[],
                lower_limits=-np.inf,
                upper_limits=np.inf,
                warm=None,
                save_fit=False,
                from_glmnet_fit=False):
    
    X = design.X
        
    exclude = np.asarray(exclude, np.int32)

    nobs, nvars = X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # compute null deviance
    # weights = weights / weights.sum()
    
    ybar = np.sum(y * weights) / np.sum(weights)
    nulldev = np.sum(weights * (y - ybar)**2)

    # if class "glmnetfit" warmstart object provided, pull whatever we want out of it
    # else, prepare arguments, then check if coefs provided as warmstart
    # (if only coefs are given as warmstart, we prepare the other arguments
    # as if no warmstart was provided)

    if isinstance(warm, ElNetWarmStart): # assumes it is a dictionary like `warm_fit`
        a = warm.a
        aint = warm.aint
        alm0 = warm.almc
        cl = warm.cl
        g = warm.g.reshape((-1,1))
        ia = warm.ia
        iy = warm.iy
        iz = warm.iz
        ju = warm.ju.reshape((-1,1))
        m = warm.m
        mm = warm.mm.reshape((-1,1))
        nino = warm.nino
        nobs = warm.no
        nvars = warm.ni
        nlp = warm.nlp
        nx = warm.nx
        r = warm.r.reshape((-1,1))
        rsqc = warm.rsqc
        xv = warm.xv.reshape((-1,1))
        vp = warm.vp.reshape((-1,1))
    else:
        
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
        r =  (weights * y).reshape((-1,1))
        rsqc = 0.                        # double(1) -- mismatch?
        xv = np.zeros((nvars, 1))        # double(nvars)

        # check if coefs were provided as warmstart: if so, use them

        if warm is not None:
            if 'beta' in warm and 'a0' in warm:
                a = np.asarray(warm['beta'], float)
                aint = np.asarray(warm['a0'], float)
                mu = X @ a + aint
                r = (weights * (y - mu)).reshape((-1,1))
                rsqc = 1 - np.sum(weights * (y - mu)**2) / nulldev
            else:
                raise ValueError('if not an instance of ElNetWarmStart `warm` should be dict-like with keys "beta" and "a0"')

    # for the parameters here, we are overriding the values provided by the
    # warmstart object

    alpha = float(alpha)                            # as.double(alpha)
    almc = float(lambda_val)                        # as.double(lambda)
    intr = int(intercept)                           # as.integer(intercept)
    jerr = 0                                        # integer(1) -- mismatch?
    maxit = int(maxit)                              # as.integer(maxit)
    thr = float(thresh)                             # as.double(thresh)
    v = np.asarray(weights, float).reshape((-1,1))  # as.double(weights)

    a_new = a.copy()

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
             'a':a_new,
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

    _args.update(**design._wls_args())

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

def _check_and_set_vp(spec, nvars):

    (penalty_factor,
     exclude) = (spec.penalty_factor,
                 spec.exclude)

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

    spec.exclude, spec.vp = exclude, vp

def _get_design(X, weights, standardize=False):
    if isinstance(X, Design):
        return X
    else:
        if weights is None:
            weights = np.ones(X.shape[0])
        return Design(X, weights, standardize=standardize)

def _wls_args(spec,
              design,
              y,
              weights,
              warm=None):

    return _elnet_args(design,
                       y,
                       weights,
                       spec.lambda_val,
                       spec.vp,
                       alpha=spec.alpha,
                       intercept=spec.intercept,
                       penalty_factor=spec.penalty_factor,
                       exclude=spec.exclude,
                       lower_limits=spec.lower_limits,
                       upper_limits=spec.upper_limits,
                       thresh=spec.control.thresh,
                       maxit=spec.control.maxit,
                       warm=warm)
