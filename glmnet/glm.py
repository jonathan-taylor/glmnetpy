from dataclasses import dataclass, asdict, field
from functools import partial
import warnings
   
import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dbn
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.linear_model import LinearRegression

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
import statsmodels.api as sm

from ._utils import (_jerr_elnetfit,
                     _obj_function,
                     _dev_function,
                     _parent_dataclass_from_child)

from .base import (Options,
                   Design)
from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)

from .elnet import (ElNetResult,
                    Design,
                    ElNetEstimator,
                    ElNetControl,
                    _check_and_set_limits,
                    _check_and_set_vp,
                    _get_design)

@add_dataclass_docstring
@dataclass
class GLMControl(ElNetControl):

    mxitnr: int = 25
    epsnr: float = 1e-6

@dataclass
class GLMMixin(Options):

    offset: np.ndarray = None
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMControl = field(default_factory=GLMControl)
add_dataclass_docstring(GLMMixin, subs={'control':'control_glm'})

@dataclass
class GLMSpec(GLMMixin, Options):
    pass
add_dataclass_docstring(GLMSpec, subs={'control':'control_glm'})

@dataclass
class GLMEstimator(BaseEstimator,
                   GLMSpec):

    def fit(self,
            X,
            y,
            sample_weight=None,
            warm=None,
            exclude=[]):

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

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        nobs, nvars = n, p = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(nobs)

        design = _get_design(X, sample_weight)
        
        nulldev = np.inf
        if not warm:
            coefold = np.zeros(nvars)   # initial coefs = 0
            intold = self.family.link((y * sample_weight).sum() / sample_weight.sum())
            mu = np.ones_like(y) * intold
            eta = self.family.link(mu)
        else:
            fit = warm
            if 'warm_fit' in warm:
                coefold = fit.warm_fit.a   # prev value for coefficients
                intold = fit.warm_fit.aint    # prev value for intercept
            elif 'intercept_' in warm and 'coef_' in warm:
                coefold = warm['coef_']   # prev value for coefficients
                intold = warm['intercept_']      # prev value for intercept
            else:
                raise ValueError("Invalid warm start object")

        state = GLMState(coef=coefold,
                         intercept=intold)

        state.update(design,
                     self.family,
                     self.offset)

        # this is just a WLS solver
        wls_est = ElNetEstimator(lambda_val=0.,
                                 fit_intercept=self.fit_intercept,
                                 standardize=False)

        def obj_function(y, family, vp, state):
            return _obj_function(y,
                                 state.mu,
                                 sample_weight,
                                 family,
                                 0,
                                 1,
                                 state.coef,
                                 vp)
        obj_function = partial(obj_function, y, self.family, np.ones(nvar))
        
        def _fit(wls_est, exclude, X, y, sample_weight):
            return wls_est.fit(X, y, sample_weight, exclude=exclude)

        _fit = partial(_fit, wls_est, exclude)
        
        (fit,
         converged,
         boundary,
         state,
         final_weights) = _IRLS(self,
                                design,
                                y,
                                sample_weight,
                                state,
                                _fit,
                                obj_function)

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
        
        if isinstance(self.family, sm_family.Gaussian):
            self.dispersion_ = _dev / (n-p-1) # usual estimate of sigma^2
        else:
            self.dispersion_ = 1

        self.unscaled_precision_ = design.quadratic_form(final_weights)
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

Returns
-------

self: object
        GLMEstimator class instance.
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




    # non-sklearn methods

    def summarize(self, dispersion=None):

        dispersion = dispersion or self.dispersion_

        keep = np.ones(self.unscaled_precision_.shape[0]-1, bool)
        if self.exclude_ is not []:
            keep[self.exclude_] = 0
        keep = np.hstack([self.fit_intercept, keep]).astype(bool)
        self.covariance_ = dispersion * np.linalg.inv(self.unscaled_precision_[keep][:,keep])
        SE = np.sqrt(np.diag(self.covariance_)) 

        index = self.feature_names_in_
        if self.fit_intercept:
            coef = np.hstack([self.intercept_, self.coef_])
            T = np.hstack([self.intercept_ / SE[0], self.coef_ / SE[1:]])
            index = ['intercept'] + index
        else:
            coef = self.coef_
            T = self.coef_ / SE

        return pd.DataFrame({'coef':coef,
                             'std err': SE,
                             't': T,
                             'P>|t|': 2 * normal_dbn.sf(np.fabs(T))},
                            index=index)
    


add_dataclass_docstring(GLMEstimator, subs={'control':'control_glm'})
    
    # # private methods

    # def _get_start(self):

    #     (design,
    #      y,
    #      weights,
    #      family,
    #      intercept,
    #      is_offset,
    #      offset,
    #      exclude,
    #      vp,
    #      alpha) = (self.design,
    #                self.y,
    #                self.weights,
    #                self.family,
    #                self.intercept,
    #                self.is_offset,
    #                self.offset,
    #                self.exclude,
    #                self.vp,
    #                self.alpha)

    #     X = design.X
    #     nobs, nvars = X.shape

    #     # compute mu and null deviance
    #     # family = binomial() gives us warnings due to non-integer weights
    #     # to avoid, suppress warnings
    #     if intercept:
    #         if is_offset:
    #             fit = sm.GLM(y,
    #                          np.ones((y.shape,1)),
    #                          family,
    #                          offset=offset,
    #                          var_weights=weights)
    #             mu = fit.fitted
    #         else:
    #             mu = np.ones(y.shape) * (weights * y).sum() / weights.sum()

    #     else:
    #         mu = family.link.inverse(offset)

    #     nulldev = _dev_function(y, mu, weights, family)

    #     # if some penalty factors are zero, we have to recompute mu

    #     vp_zero = sorted(set(exclude).difference(np.nonzero(vp == 0)[0]))
    #     if vp_zero:
    #         tempX = X[:,vp_zero]

    #         if scipy.sparse.issparse(X):
    #             tempX = X.toarray()

    #         if intercept:
    #             tempX = sm.add_constant(tempX)

    #         tempfit = sm.GLM(y,
    #                          tempX,
    #                          family,
    #                          offset=offset,
    #                          var_weights=weights)
    #         mu = tempfit.fittedvalues

    #     # compute lambda max
    #     ju = np.ones(nvars)
    #     ju[exclude] = 0 # we have already included constant variables in exclude

    #     r = y - mu
    #     eta = family.link(mu)
    #     v = family.variance(mu)
    #     m_e = family.link.inverse_deriv(eta)
    #     weights = weights / weights.sum()

    #     rv = r / v * m_e * weights

    #     if scipy.sparse.issparse(X):
    #         xm, xs = design.xm, design.xs
    #         g = np.abs((X.T @ rv - np.sum(rv) * xm) / xs)
    #     else:
    #         g = np.abs(X.T @ rv)

    #     g = g * ju / (vp + (vp <= 0))
    #     lambda_max = np.max(g) / max(alpha, 1e-3)

    #     return {'nulldev':nulldev,
    #             'mu':mu,
    #             'lambda_max':lambda_max}

    # def _get_initial_state(self,
    #                        warm=None):

    #     nobs, nvars = self.X.shape

    #     # get offset

    #     is_offset = self.offset is not None

    #     if not warm:
    #         start_val = self._get_start()
    #         nulldev = start_val['nulldev']
    #         mu = start_val['mu']
    #         fit = None
    #         coefold = np.zeros(nvars)   # initial coefs = 0
    #         eta = self.family.link(mu)
    #         intold = (eta - self.offset)[0]

    #     else:
    #         fit = warm
    #         if 'warm_fit' in warm:
    #             nulldev = fit['nulldev']
    #             coefold = fit.warm_fit.a   # prev value for coefficients
    #             intold = fit.warm_fit.aint    # prev value for intercept
    #         elif 'a0' in warm and 'beta' in warm:
    #             nulldev = self._get_start()['nulldev']

    #             coefold = warm['beta']   # prev value for coefficients
    #             intold = warm['a0']      # prev value for intercept
    #         else:
    #             raise ValueError("Invalid warm start object")

    #     state = GLMState(coef=coefold,
    #                      intercept=intold)
    #     state.update(self.design,
    #                  self.family,
    #                  self.offset)

    #     return state, nulldev


@add_dataclass_docstring
@dataclass
class GLMResult(ElNetResult):

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

def _quasi_newton_step(spec,
                       design,
                       y,
                       weights,
                       state,
                       elnet_solver,
                       objective,
                       fit=None):

    coefold, intold = state.coef, state.intercept

    # some checks for NAs/zeros
    varmu = spec.family.variance(state.mu)
    if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

    if np.any(varmu == 0): raise ValueError("0s in V(mu)")

    dmu_deta = spec.family.link.inverse_deriv(state.eta)
    if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

    # compute working response and weights
    if spec.offset is not None:
        z = (state.eta - spec.offset) + (y - state.mu) / dmu_deta
    else:
        z = state.eta + (y - state.mu) / dmu_deta
    
    w = (weights * dmu_deta**2)/varmu

    # have to update the weighted residual in our fit object
    # (in theory g and iy should be updated too, but we actually recompute g
    # and it anyway in wls.f)

    if fit is not None:
        if spec.offset is not None:
            fit.warm_fit.r = w * (z - state.eta + spec.offset)
        else:
            fit.warm_fit.r = w * (z - state.eta)
        warm = fit.warm_fit
    else:
        warm = None

    # should maybe do smarter with sparse scipy.linalg.LinearOperator?
    
    lm = LinearRegression(fit_intercept=spec.fit_intercept)
    lm.fit(design.X, z, sample_weight=w)
    
    coefnew = lm.coef_
    intnew = lm.intercept_
    
    state = GLMState(coefnew,
                     intnew,
                     obj_val_old=state.obj_val)

    state.update(design,
                 spec.family,
                 spec.offset)

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
        return np.isfinite(state.obj_val) and state.obj_val < spec.control.big, boundary, halved

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
                if ii > spec.control.mxitnr:
                    raise ValueError(f"inner loop {test}; cannot correct step size")
                ii += 1

                state = GLMState((state.coef + coefold)/2,
                                 (state.intercept + intold)/2,
                                 obj_val_old=state.obj_val_old)
                state.update(design,
                             spec.family,
                             spec.offset)
                state.obj_val = objective(state)
                check, boundary_, halved_ = test(state)

    return state, fit, boundary, halved

def _IRLS(spec,
          design,
          y,
          weights,
          state,
          elnet_solver,
          objective):

    coefold, intold = state.coef, state.intercept

    state.obj_val_old = objective(state)
    converged = False
    fit = None

    for _ in range(spec.control.mxitnr):

        (state,
         fit,
         boundary,
         halved) = _quasi_newton_step(spec,
                                      design,
                                      y,
                                      weights,
                                      state,
                                      elnet_solver,
                                      objective,
                                      fit=fit)

        # test for convergence
        if (np.fabs(state.obj_val - state.obj_val_old)/(0.1 + abs(state.obj_val)) < spec.control.epsnr):
            converged = True
            break

    return fit, converged, boundary, state, weights
