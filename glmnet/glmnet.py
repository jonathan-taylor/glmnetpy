from dataclasses import dataclass, asdict, field, InitVar
from functools import partial
import warnings
   
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import scipy.sparse
from scipy.stats import norm as normal_dbn

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.linear_model import LinearRegression

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
import statsmodels.api as sm

from ._utils import (_obj_function,
                     _dev_function,
                     _parent_dataclass_from_child)

from .base import Design, _get_design, Penalty
from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)

from .elnet import ElNetEstimator, ElNetControl, ElNetSpec
from .glm import GLMState, _IRLS

@add_dataclass_docstring
@dataclass
class GLMNetControl(ElNetControl):

    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35

@dataclass
class GLMNetSpec(ElNetSpec):

    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMNetControl = field(default_factory=GLMNetControl)

add_dataclass_docstring(GLMNetSpec, subs={'control':'control_glmnet'})

@add_dataclass_docstring
@dataclass
class GLMNetResult(object):

    family: sm_family.Family
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float

@dataclass
class GLMState(object):

    coef: np.ndarray
    intercept: np.ndarray
    obj_val: float = np.inf
    obj_val_old: float = np.inf
    
    def update(self,
               design,
               family,
               offset):
        '''pin the mu/eta values to coef/intercept'''
        self.eta = design.linear_map(self.coef,
                                     self.intercept)
        if offset is None:
            self.mu = family.link.inverse(self.eta)
        else:
            self.mu = family.link.inverse(self.eta + offset)    

@dataclass
class GLMNetRegularizer(Penalty):

    fit_intercept: bool = False
    warm_fit: dict = field(default_factory=dict)
    nvars: InitVar[int] = None
    control: InitVar[GLMNetControl] = None
    
    def __post_init__(self, nvars, control):

        if self.lower_limits == -np.inf:
            self.lower_limits = -np.ones(nvars) * control.big
        
        if self.upper_limits == np.inf:
            self.upper_limits = np.ones(nvars) * control.big
        
        if self.penalty_factor is None:
            self.penalty_factor = np.ones(nvars)

        self.penalty_factor *= nvars / self.penalty_factor.sum() 
            
        self.elnet_estimator = ElNetEstimator(lambda_val=self.lambda_val,
                                              alpha=self.alpha,
                                              control=control,
                                              lower_limits=self.lower_limits,
                                              upper_limits=self.upper_limits,
                                              fit_intercept=self.fit_intercept,
                                              standardize=False)

    def quasi_newton_step(self,
                          design,
                          pseudo_response,
                          sample_weight):

        z = pseudo_response
        w = sample_weight

        out = self.elnet_estimator.fit(design.X, z, sample_weight=sample_weight).result_
        coefnew = out.beta.toarray().reshape(-1)
        intnew = out.a0
        
        self.warm_fit['coef_'] = coefnew
        self.warm_fit['intercept_'] = intnew
        
        return coefnew, intnew

    def get_warm_start(self):

        if ('coef_' in self.warm_fit.keys() and
            'intercept_' in self.warm_fit_keys()):

            return (self.warm_fit['coef_'],
                    self.warm_fit['intercept_']) 

    def update_resid(self, r):
        self.warm_fit['resid_'] = r
        
    def objective(self, state):
        return 0
# end of GLMNetRegularizer

@dataclass
class GLMNetEstimator(BaseEstimator,
                      GLMNetSpec):

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            dispersion=1,
            offset=None):

        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        self.exclude_ = exclude
        nobs, nvar = X.shape
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        X, y = check_X_y(X, y,
                         accept_sparse=['csc'],
                         multi_output=False,
                         estimator=self)

        y = np.asarray(y)

        if regularizer is None:
            regularizer = GLMNetRegularizer(lambda_val=self.lambda_val,
                                            penalty_factor=self.penalty_factor,
                                            lower_limits=self.lower_limits,
                                            upper_limits=self.upper_limits,
                                            fit_intercept=self.fit_intercept,
                                            nvars=X.shape[1],
                                            control=self.control)
        self.regularizer_ = regularizer

        if self.control is None:
            self.control = GLMNetControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMNetControl,
                                                        self.control)
        nobs, nvars = n, p = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(nobs)

        design = _get_design(X, sample_weight)
        self.design_ = design
        
        nulldev = np.inf

        warm_start = self.regularizer_.get_warm_start()
        if warm_start is None:
            coefold = np.zeros(nvars)   # initial coefs = 0
            intold = self.family.link((y * sample_weight).sum() / sample_weight.sum())
        else:
            coefold, intold = warm_start

        mu = np.ones_like(y) * intold
        eta = self.family.link(mu)

        state = GLMState(coef=coefold,
                         intercept=intold)

        state.update(design,
                     self.family,
                     offset)

        def obj_function(y, family, regularizer, state):
            return (_dev_function(y,
                                 state.mu,
                                 sample_weight,
                                 family) +
                    regularizer.objective(state))
        obj_function = partial(obj_function, y, self.family, regularizer)
        
        (converged,
         boundary,
         state,
         final_weights) = _IRLS(regularizer,
                                self.family,
                                design,
                                y,
                                offset,
                                sample_weight,
                                state,
                                obj_function,
                                self.control)

        # checks on convergence and fitted values
        if not converged:
            warnings.warn("fitting IRLS: algorithm did not converge")
        if boundary:
            warnings.warn("fitting IRLS: algorithm stopped at boundary value")

        _dev = _dev_function(y,
                             state.mu,
                             sample_weight,
                             self.family)

        self.coef_ = state.coef
        self.intercept_ = state.intercept
        
        return self
    fit.__doc__ = '''
Fit a GLMNet.

Parameters
----------

{X}
{y}
{weights}
{warm_glm}
{exclude}
{summarize}
{offset}
    
Returns
-------

self: object
        GLMNetEstimator class instance.
        '''.format(**_docstrings)
    
    def predict(self, X, prediction_type='mean'):

        eta = X @ self.coef_ + self.intercept_
        if prediction_type == 'linear':
            return eta
        elif prediction_type == 'mean':
            return self.family.link.inverse(eta)
        else:
            raise ValueError("prediction should be one of 'mean' or 'linear'")
    predict.__doc__ = '''
Predict outcome of corresponding family.

Parameters
----------

{X}
{prediction_type}

Returns
-------

{prediction}'''.format(**_docstrings).strip()

    def score(self, X, y, sample_weight=None):

        mu = self.predict(X, prediction_type='mean')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return -_dev_function(y, mu, sample_weight, self.family) / 2 
    score.__doc__ = '''
Compute log-likelihood (i.e. negative deviance / 2) for test X and y using fitted model.

Parameters
----------

{X}
{y}
{sample_weight}    

Returns
-------

score: float
    Deviance of family for (X, y).
'''.format(**_docstrings).strip()

add_dataclass_docstring(GLMNetEstimator, subs={'control':'control_glm'})

