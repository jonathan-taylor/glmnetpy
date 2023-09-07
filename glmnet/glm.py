from dataclasses import dataclass, asdict, field
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

from ._utils import (_dev_function,
                     _parent_dataclass_from_child)

from .base import Design, _get_design
from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)

from .elnet import ElNet

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
    obj_val_old: float = np.inf
    
    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset):
        '''pin the mu/eta values to coef/intercept'''
        self.eta = design @ self._stack
        if offset is None:
            self.mu = family.link.inverse(self.eta)
        else:
            self.mu = family.link.inverse(self.eta + offset)    

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

    def quasi_newton_step(self,
                          design,
                          pseudo_response,
                          sample_weight):

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
                    warnings.warn("error in solve: possible singular matrix, trying pseudo-inverse")
                    beta = np.linalg.pinv(XW) @ Wz
                coefnew = beta[1:]
                intnew = beta[0]

            else:
                coefnew = np.linalg.pinv(XW) @ (sqrt_w * z)
                intnew = 0

        self.warm_state['coef_'] = coefnew
        self.warm_state['intercept_'] = intnew
        
        return coefnew, intnew

    def get_warm_start(self):

        if ('coef_' in self.warm_state.keys() and
            'intercept_' in self.warm_state_keys()):

            state = GLMState(self.warm_state['coef_'],
                             self.warm_state['intercept_'])
            return state

    def update_resid(self, r):
        self.warm_state['resid_'] = r
        
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
            offset=None):

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

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        nobs, nvars = n, p = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(nobs) 
        self.sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        design = self._get_design(X, normed_sample_weight)
        self.design_ = design
        
        if self.fit_intercept:
            mu0 = (y * normed_sample_weight).sum() * np.ones_like(y)
        else:
            mu0 = self.family.link.inverse(np.zeros(y.shape, float))
        self.null_deviance_ = _dev_function(y,
                                            mu0,
                                            sample_weight, # not normed_sample_weight!
                                            self.family)

        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(X)
        self.regularizer_ = regularizer

        state = self.regularizer_.get_warm_start()

        if state is None:
            coefold = np.zeros(nvars)   # initial coefs = 0
            intold = self.family.link(mu0[0])
            state = GLMState(coef=coefold,
                             intercept=intold)

        state.update(design,
                     self.family,
                     offset)

        def obj_function(y, family, regularizer, state):
            return (_dev_function(y,
                                  state.mu,
                                  normed_sample_weight,
                                  family) / 2 +
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
                                normed_sample_weight,
                                state,
                                obj_function,
                                self.control)

        # checks on convergence and fitted values
        if not converged:
            warnings.warn("fitting IRLS: algorithm did not converge")
        if boundary:
            warnings.warn("fitting IRLS: algorithm stopped at boundary value")

        self.deviance_ = _dev_function(y,
                                       state.mu,
                                       sample_weight, # not the normalized weights!
                                       self.family)

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

        mu = self.predict(X, prediction_type='mean')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return -_dev_function(y, mu, sample_weight, self.family) / 2 
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
        self.coef_ = state.coef
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



# private functions

def _quasi_newton_step(regularizer,
                       family,
                       design,
                       y,
                       offset,
                       weights,
                       state,
                       objective,
                       control):

    coefold, intold = state.coef, state.intercept

    # some checks for NAs/zeros
    varmu = family.variance(state.mu)
    if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

    if np.any(varmu == 0): raise ValueError("0s in V(mu)")

    dmu_deta = family.link.inverse_deriv(state.eta)
    if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

    # compute working response and weights
    if offset is not None:
        z = (state.eta - offset) + (y - state.mu) / dmu_deta
    else:
        z = state.eta + (y - state.mu) / dmu_deta
    
    newton_weights = w = (weights * dmu_deta**2)/varmu

    # could have the quasi_newton_step return state instead?
    
    coefnew, intnew = regularizer.quasi_newton_step(design, z, w)
    state = GLMState(coefnew,
                     intnew,
                     obj_val_old=state.obj_val)

    state.update(design,
                 family,
                 offset)

    state.obj_val = objective(state)

    # check to make sure it is a feasible descent step

    boundary = False
    halved = False  # did we have to halve the step size?

    # three checks we'll apply

    # FIX THESE 
    valideta = lambda eta: True
    validmu = lambda mu: True

    # not sure boundary / halved handled correctly

    def finite_objective(state):
        boundary = True
        halved = True
        return np.isfinite(state.obj_val) and state.obj_val < control.big, boundary, halved

    def valid(state):
        boundary = True
        halved = True
        return valideta(state.eta) and validmu(state.mu), boundary, halved

    def decreased_obj(state):
        boundary = False
        halved = True

        return state.obj_val <= state.obj_val_old + 1e-7, boundary, halved

    for test, msg in [(finite_objective,
                       "Non finite objective function! Step size truncated due to divergence."),
                      (valid,
                       "Invalid eta/mu! Step size truncated: out of bounds."),
                      (decreased_obj,
                       "")]:

        if not test(state)[0]:
            if msg:
                warnings.warn(msg)
            if np.any(np.isnan(coefold)) or np.isnan(intold):
                raise ValueError("No valid set of coefficients has been found: please supply starting values")

            ii = 1
            check, boundary_, halved_ = test(state)
            if not check:
                boundary = boundary or boundary_
                halved = halved or halved_

            while not check:
                if ii > control.mxitnr:
                    raise ValueError(f"inner loop {test}; cannot correct step size")
                ii += 1

                state = GLMState((state.coef + coefold)/2,
                                 (state.intercept + intold)/2,
                                 obj_val_old=state.obj_val_old)
                state.update(design,
                             family,
                             offset)
                state.obj_val = objective(state)
                check, boundary_, halved_ = test(state)

    if offset is not None:
        regularizer.update_resid(w * (z - state.eta + offset))
    else:
        regularizer.update_resid(w * (z - state.eta))

    return state, boundary, halved, newton_weights

def _IRLS(regularizer,
          family,
          design,
          y,
          offset,
          weights,
          state,
          objective,
          control):

    coefold, intold = state.coef, state.intercept

    state.obj_val_old = objective(state)
    converged = False

    for _ in range(control.mxitnr):

        (state,
         boundary,
         halved,
         newton_weights) = _quasi_newton_step(regularizer,
                                              family,
                                              design,
                                              y,
                                              offset,
                                              weights,
                                              state,
                                              objective,
                                              control)

        # test for convergence
        if (np.fabs(state.obj_val - state.obj_val_old)/(0.1 + abs(state.obj_val)) < control.epsnr):
            converged = True
            break

    return converged, boundary, state, newton_weights
