from dataclasses import dataclass, asdict, field
from functools import partial
import logging
LOG = False
   
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import scipy.sparse
from scipy.stats import norm as normal_dbn

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.linear_model import LinearRegression

from statsmodels.genmod.families import family as sm_family

from ._utils import _parent_dataclass_from_child

from .base import Design, _get_design
from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)
from .irls import IRLS

@add_dataclass_docstring
@dataclass
class GLMControl(object):

    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35

@dataclass
class GLMSpec(object):

    fit_intercept: bool = True
    summarize: bool = False
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMControl = field(default_factory=GLMControl)

add_dataclass_docstring(GLMSpec, subs={'control':'control_glm'})

@add_dataclass_docstring
@dataclass
class GLMResult(object):

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
    pmin: float = 1e-9
    
    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset,
               objective=None):
        '''pin the mu/eta values to coef/intercept'''
        self.eta = design @ self._stack
        if offset is None:
            self.linear_predictor = self.eta
        else:
            self.linear_predictor = self.eta + offset
        self.mu = family.link.inverse(self.linear_predictor)
        if isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.linear_predictor = family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y):

        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.eta)
        
        # compute working residual
        r = (y - self.mu) 
        return r / varmu * dmu_deta

@dataclass
class GLMRegularizer(object):

    fit_intercept: bool = False
    warm_state: dict = field(default_factory=dict)

    def half_step(self,
                  state,
                  oldstate):
        return GLMState(0.5 * (oldstate.coef + state.coef),
                        0.5 * (oldstate.intercept + state.intercept))

    def _debug_msg(self,
                   state):
        return f'Coef: {state.coef}, Intercept: {state.intercept}, Objective: {state.obj_val}'

    def check_state(self,
                    state):
        if np.any(np.isnan(state.coef)):
            raise ValueError('coef has NaNs')
        if np.isnan(state.intercept):
            raise ValueError('intercept is NaN')

    def newton_step(self,
                    design,
                    pseudo_response,
                    sample_weight,
                    cur_state):   # ignored for GLM

        z = pseudo_response
        w = sample_weight

        if scipy.sparse.issparse(design.X):
            lm = LinearRegression(fit_intercept=self.fit_intercept)
            lm.fit(design.X, z, sample_weight=w)
            coefnew = lm.coef_
            intnew = lm.intercept_

        else:
            sqrt_w = np.sqrt(w)
            XW = design.X * sqrt_w[:, None]
            if self.fit_intercept:
                Wz = sqrt_w * z
                XW = np.concatenate([sqrt_w.reshape((-1,1)), XW], axis=1)
                Q = XW.T @ XW
                V = XW.T @ Wz
                try:
                    beta = np.linalg.solve(Q, V)
                except LinAlgError as e:
                    if LOG: logging.debug("Error in solve: possible singular matrix, trying pseudo-inverse")
                    beta = np.linalg.pinv(XW) @ Wz
                coefnew = beta[1:]
                intnew = beta[0]

            else:
                coefnew = np.linalg.pinv(XW) @ (sqrt_w * z)
                intnew = 0

        self.warm_state = GLMState(coefnew,
                                   intnew)
        
        return self.warm_state

    def objective(self, state):
        return 0

add_dataclass_docstring(GLMRegularizer, subs={'warm_state':'warm_glm'})
# end of GLMRegularizer

@dataclass
class GLM(BaseEstimator,
          GLMSpec):

    def _get_regularizer(self,
                         X):
        return GLMRegularizer(fit_intercept=self.fit_intercept)

    # no standardization for GLM
    def _get_design(self,
                    X,
                    sample_weight):
        return _get_design(X, sample_weight)

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            dispersion=1,
            offset=None,
            check=True):

        nobs, nvar = X.shape
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        if check:
            X, y = check_X_y(X, y,
                             accept_sparse=['csc'],
                             multi_output=False,
                             estimator=self)

        y = np.asarray(y)

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        nobs, nvars = n, p = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(nobs) 
        self.sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        if not hasattr(self, "design_"):
            self.design_ = design = self._get_design(X, normed_sample_weight)
        else:
            design = self.design_
        
        if self.fit_intercept:
            mu0 = (y * normed_sample_weight).sum() * np.ones_like(y)
        else:
            mu0 = self.family.link.inverse(np.zeros(y.shape, float))
        self.null_deviance_ = self.family.deviance(y, mu0, freq_weights=sample_weight)

        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(X)
        self.regularizer_ = regularizer

        if (hasattr(self.regularizer_, 'warm_state') and
            self.regularizer_.warm_state):
            state = self.regularizer_.warm_state
        else:
            coefold = np.zeros(nvars)   # initial coefs = 0
            intold = self.family.link(mu0[0])
            state = GLMState(coef=coefold,
                             intercept=intold)

        def obj_function(y, normed_sample_weight, family, regularizer, state):
            val1 = family.deviance(y, state.mu, freq_weights=normed_sample_weight) / 2
            val2 = regularizer.objective(state)
            val = val1 + val2
            if LOG: logging.debug(f'Computing objective, lambda: {regularizer.lambda_val}, alpha: {regularizer.alpha}, coef: {state.coef}, intercept: {state.intercept}, deviance: {val1}, penalty: {val2}')
            return val
        obj_function = partial(obj_function, y.copy(), normed_sample_weight.copy(), self.family, regularizer)
        
        state.update(design,
                     self.family,
                     offset,
                     obj_function)

        (converged,
         boundary,
         state,
         final_weights) = IRLS(regularizer,
                               self.family,
                               design,
                               y,
                               offset,
                               normed_sample_weight,
                               state,
                               obj_function,
                               self.control)

        # checks on convergence and fitted values
        if not converged:
            if LOG: logging.debug("Fitting IRLS: algorithm did not converge")
        if boundary:
            if LOG: logging.debug("Fitting IRLS: algorithm stopped at boundary value")


        self.deviance_ = self.family.deviance(y,
                                              state.mu,
                                              freq_weights=sample_weight) # not the normalized weights!

        self._set_coef_intercept(state)

        if isinstance(self.family, sm_family.Gaussian):
            self.dispersion_ = self.deviance_ / (n-p-self.fit_intercept) # usual estimate of sigma^2
        else:
            self.dispersion_ = dispersion

        if self.summarize:
            unscaled_precision_ = design.quadratic_form(final_weights * sample_weight.sum()) # IRLS used normalized weights,
                                                                                             # this unnormalizes them...

            keep = np.ones(unscaled_precision_.shape[0]-1, bool)
            if exclude is not []:
                keep[exclude] = 0
            keep = np.hstack([self.fit_intercept, keep]).astype(bool)
            self.covariance_ = dispersion * np.linalg.inv(unscaled_precision_[keep][:,keep])

            SE = np.sqrt(np.diag(self.covariance_)) 
            index = self.feature_names_in_
            if self.fit_intercept:
                coef = np.hstack([self.intercept_, self.coef_])
                T = np.hstack([self.intercept_ / SE[0], self.coef_ / SE[1:]])
                index = ['intercept'] + index
            else:
                coef = self.coef_
                T = self.coef_ / SE

            self.summary_ = pd.DataFrame({'coef':coef,
                                          'std err': SE,
                                          't': T,
                                          'P>|t|': 2 * normal_dbn.sf(np.fabs(T))},
                                         index=index)
        else:
            self.summary_ = self.covariance_ = None
            
        return self
    fit.__doc__ = '''
Fit a GLM.

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
        GLM class instance.
        '''.format(**_docstrings)
    
    def predict(self, X, prediction_type='response'):

        eta = X @ self.coef_ + self.intercept_
        if prediction_type == 'link':
            return eta
        elif prediction_type == 'response':
            return self.family.link.inverse(eta)
        else:
            raise ValueError("prediction should be one of 'response' or 'link'")
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

        mu = self.predict(X, prediction_type='response')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return -family.deviance(y, mu, freq_weights=sample_weight) / 2
    score.__doc__ = '''
Compute weighted log-likelihood (i.e. negative deviance / 2) for test X and y using fitted model. Weights
default to `np.ones_like(y) / y.shape[0]`.

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

    def _set_coef_intercept(self, state):
        self.coef_ = state.coef.copy() # this makes a copy -- important to make this copy because `state.coef` is persistent
        if hasattr(self, 'standardize') and self.standardize:
            self.scaling_ = self.design_.scaling_
            self.coef_ /= self.scaling_ 
        self.intercept_ = state.intercept - (self.coef_ * self.design_.centers_).sum()

GLM.__doc__ = '''
Class to fit a Generalized Linear Model (GLM). Base class for `GLMNet`.

Parameters
----------
{fit_intercept}
{summarize}
{family}
{control_glm}

Attributes
__________
{coef_}
{intercept_}
{summary_}
{covariance_}
{null_deviance_}
{deviance_}
{dispersion_}
{regularizer_}
'''.format(**_docstrings)



