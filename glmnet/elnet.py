import logging

from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y

from ._elnet_point import elnet_point as dense_wls
from ._elnet_point import spelnet_point as sparse_wls
from .base import (Base,
                   Penalty,
                   Design,
                   _get_design)
from ._utils import _jerr_elnetfit
from .docstrings import (make_docstring,
                         add_dataclass_docstring)

@dataclass
class ElNetControl(object):
    """Control parameters for ElNet optimization.
    
    Parameters
    ----------
    thresh : float, default=1e-7
        Convergence threshold for optimization.
    maxit : int, default=100000
        Maximum number of iterations.
    big : float, default=9.9e35
        Large number used for bounds.
    logging : bool, default=False
        Whether to enable debug logging.
    """

    thresh: float = 1e-7
    maxit: int = 100000
    big: float = 9.9e35
    logging: bool = False

@dataclass
class ElNetSpec(Penalty):
    """Specification for ElNet model parameters.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    standardize : bool, default=True
        Whether to standardize features.
    control : ElNetControl, optional
        Control parameters for optimization.
    exclude : list, default_factory=list
        List of variable indices to exclude from penalization.
    """

    fit_intercept: bool = True
    standardize: bool = True
    control: ElNetControl = field(default_factory=ElNetControl)
    exclude: list = field(default_factory=list)

add_dataclass_docstring(ElNetSpec, subs={'control':'control_elnet'})

@dataclass
class ElNet(BaseEstimator,
            RegressorMixin,
            ElNetSpec):
    """Elastic Net regression model.
    
    This class implements elastic net regression using coordinate descent.
    It supports both dense and sparse input matrices.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing parameter. alpha=1 is lasso, alpha=0 is ridge.
    lambda_val : float, default=0.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    standardize : bool, default=True
        Whether to standardize features.
    penalty_factor : array-like, optional
        Multiplicative factors for penalty for each coefficient.
    lower_limits : array-like, optional
        Lower bounds for coefficients.
    upper_limits : array-like, optional
        Upper bounds for coefficients.
    control : ElNetControl, optional
        Control parameters for optimization.
    exclude : list, default_factory=list
        List of variable indices to exclude from penalization.
    """

    def fit(self,
            X,
            y,
            sample_weight=None,
            warm=None,
            check=True):
        """Fit the elastic net model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        warm : tuple, optional
            Warm start parameters (coef, intercept, eta).
        check : bool, default=True
            Whether to perform input validation.
            
        Returns
        -------
        self : object
            Returns self.
        """

        if not hasattr(self, "design_"):
            self.design_ = design = _get_design(X,
                                                sample_weight,
                                                standardize=self.standardize,
                                                intercept=self.fit_intercept)
        else:
            design = self.design_

        if self.lambda_val > 0 or not (np.all(design.centers_ == 0) and np.all(design.scaling_ == 1)):

            if self.control is None:
                self.control = ElNetControl()

            nobs, nvars = design.X.shape

            # because _get_design ignores `standardize` if X is a `Design`, then if `X`
            # is a `Design` this will ignore `self.standardize

            design.X, y = check_X_y(design.X, y,
                                    accept_sparse=['csc'],
                                    multi_output=False,
                                    estimator=self)

            if sample_weight is None:
                sample_weight = np.ones(nobs) / nobs

            _check_and_set_limits(self, nvars)
            self.exclude = _check_and_set_vp(self, nvars, self.exclude)

            if hasattr(self, "_wrapper_args") and warm is not None:
                args = self._wrapper_args
                nulldev = self._nulldev
                
                coef, intercept, eta = warm

                # some sanity checks to see if design matches
                
                if ((design.shape[1] != coef.shape[0] + 1) or
                    (design.shape[0] != eta.shape[0])): 
                    raise ValueError('dimension mismatch for warm start and design')

                # set the warm start, copying into their arrays will check that sizes are correct

                args['aint'] = intercept
                args['a'][:] = coef.reshape((-1,1))
                args['r'][:] = (sample_weight * (y - eta)).reshape((-1,1))

            else:
                args, nulldev = _elnet_wrapper_args(design,
                                                    y,
                                                    sample_weight,
                                                    self.lambda_val,
                                                    alpha=self.alpha,
                                                    intercept=self.fit_intercept,
                                                    penalty_factor=self.penalty_factor,
                                                    exclude=self.exclude,
                                                    lower_limits=self.lower_limits,
                                                    upper_limits=self.upper_limits,
                                                    thresh=self.control.thresh,
                                                    maxit=self.control.maxit)
                self._wrapper_args = args
                self._nulldev = nulldev
                    
            # override to current values

            args['alpha'] = float(self.alpha)                                 # as.double(alpha)
            args['almc'] = float(self.lambda_val)                             # as.double(lambda)
            args['intr'] = int(self.fit_intercept)                            # as.integer(intercept)
            args['jerr'] = 0                                                  # integer(1) -- mismatch?
            args['maxit'] = int(self.control.maxit)                           # as.integer(maxit)
            args['thr'] = float(self.control.thresh)                          # as.double(thresh)
            args['v'] = np.asarray(sample_weight, float).reshape((-1,1))      # as.double(weights)

            if self.control.logging: logging.debug(f'Elnet warm coef: {args["a"]}, Elnet warm intercept: {args["aint"]}')

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
                if self.control.logging: logging.debug(errmsg['msg'])

            if self.control.logging: logging.debug(f'Elnet coef: {wls_fit["a"]}, Elnet intercept: {wls_fit["aint"]}')

            self.raw_coef_ = wls_fit['a'].reshape(-1)
            self.raw_intercept_ = wls_fit['aint']


        else:
            # can use LinearRegression

            lm = LinearRegression(fit_intercept=self.fit_intercept)
            if scipy.sparse.issparse(design.X):
                X_s = scipy.sparse.csc_array(design.X)
            else:
                X_s = design.X
            lm.fit(X_s, y, sample_weight)
            
            self.raw_coef_ = lm.coef_
            self.raw_intercept_ = lm.intercept_

        self.design_ = design
        self.coef_ = self.raw_coef_ / design.scaling_
        self.intercept_ = self.raw_intercept_ - (self.coef_ * self.design_.centers_).sum()

        return self

add_dataclass_docstring(ElNet, subs={'control':'control_elnet'})

def _elnet_wrapper_args(design,
                        y,
                        sample_weight,
                        lambda_val,
                        alpha=1.0,
                        intercept=True,
                        thresh=1e-7,
                        maxit=100000,
                        penalty_factor=None, 
                        exclude=[],
                        lower_limits=-np.inf,
                        upper_limits=np.inf):
    """Create wrapper arguments for ElNet C++ function.
    
    Parameters
    ----------
    design : Design
        Design matrix object.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Sample weights.
    lambda_val : float
        Regularization strength.
    alpha : float, default=1.0
        Elastic net mixing parameter.
    intercept : bool, default=True
        Whether to fit intercept.
    thresh : float, default=1e-7
        Convergence threshold.
    maxit : int, default=100000
        Maximum iterations.
    penalty_factor : array-like, optional
        Penalty factors for each coefficient.
    exclude : list, default=[]
        Variables to exclude from penalization.
    lower_limits : array-like, optional
        Lower bounds for coefficients.
    upper_limits : array-like, optional
        Upper bounds for coefficients.
        
    Returns
    -------
    args : dict
        Arguments for C++ function.
    nulldev : float
        Null deviance.
    """

    
    X = design.X
        
    exclude = np.asarray(exclude, np.int32)

    nobs, nvars = X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # compute null deviance
    # sample_weight = sample_weight / sample_weight.sum()
    
    ybar = np.sum(y * sample_weight) / np.sum(sample_weight)
    nulldev = np.sum(sample_weight * (y - ybar)**2)

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
             'vp':penalty_factor,
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

    _args.update(**_design_wrapper_args(design))

    return _args, nulldev

def _check_and_set_limits(spec, nvars):
    """Check and set coefficient limits.
    
    Parameters
    ----------
    spec : ElNetSpec
        Specification object.
    nvars : int
        Number of variables.
    """

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
    """Check and set penalty factors.
    
    Parameters
    ----------
    spec : ElNetSpec
        Specification object.
    nvars : int
        Number of variables.
    exclude : list
        Variables to exclude.
        
    Returns
    -------
    exclude : list
        Updated exclude list.
    """

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

def _design_wrapper_args(design):
    """Create design wrapper arguments for C++ function.
    
    Parameters
    ----------
    design : Design
        Design matrix object.
        
    Returns
    -------
    dict
        Arguments for C++ function.
    """
    if not scipy.sparse.issparse(design.X):
        return {'x':design.X}
    else:
        return {'x_data_array':design.X.data,
                'x_indices_array':design.X.indices,
                'x_indptr_array':design.X.indptr,
                'xm':design.centers_,
                'xs':design.scaling_}
