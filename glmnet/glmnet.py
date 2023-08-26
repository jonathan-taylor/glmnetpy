import warnings
from dataclasses import dataclass, asdict, field
   
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, RegressorMixin

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
import statsmodels.api as sm

from ._utils import (_jerr_elnetfit,
                     _obj_function,
                     _dev_function,
                     _parent_dataclass_from_child)

from .base import Penalty, Options, Design
from .docstrings import make_docstring, add_dataclass_docstring

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
class GLMMixin(object):

    offset: np.ndarray = None
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMControl = field(default_factory=GLMControl)
add_dataclass_docstring(GLMMixin, subs={'control':'control_glm'})

@dataclass
class GLMSpec(GLMMixin, Options):
    pass
add_dataclass_docstring(GLMSpec, subs={'control':'control_glm'})

@dataclass
class GLMEstimator(GLMMixin, ElNetEstimator):

    def fit(self, X, y, weights=None, warm=None):

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        if self.exclude is None:
            self.exclude = []
            
        nobs, nvars = X.shape
        
        if weights is None:
            weights = np.ones(nobs)

        design = _get_design(X, weights)
        _check_and_set_limits(self, nvars)
        _check_and_set_vp(self, nvars)
        
        state, self.nulldev = self._get_initial_state(warm)
        return self._fit(state)
    
    # private methods

    def _get_start(self):

        (design,
         y,
         weights,
         family,
         intercept,
         is_offset,
         offset,
         exclude,
         vp,
         alpha) = (self.design,
                   self.y,
                   self.weights,
                   self.family,
                   self.intercept,
                   self.is_offset,
                   self.offset,
                   self.exclude,
                   self.vp,
                   self.alpha)

        X = design.X
        nobs, nvars = X.shape

        # compute mu and null deviance
        # family = binomial() gives us warnings due to non-integer weights
        # to avoid, suppress warnings
        if intercept:
            if is_offset:
                fit = sm.GLM(y,
                             np.ones((y.shape,1)),
                             family,
                             offset=offset,
                             var_weights=weights)
                mu = fit.fitted
            else:
                mu = np.ones(y.shape) * (weights * y).sum() / weights.sum()

        else:
            mu = family.link.inverse(offset)

        nulldev = _dev_function(y, mu, weights, family)

        # if some penalty factors are zero, we have to recompute mu

        vp_zero = sorted(set(exclude).difference(np.nonzero(vp == 0)[0]))
        if vp_zero:
            tempX = X[:,vp_zero]

            if scipy.sparse.issparse(X):
                tempX = X.toarray()

            if intercept:
                tempX = sm.add_constant(tempX)

            tempfit = sm.GLM(y,
                             tempX,
                             family,
                             offset=offset,
                             var_weights=weights)
            mu = tempfit.fittedvalues

        # compute lambda max
        ju = np.ones(nvars)
        ju[exclude] = 0 # we have already included constant variables in exclude

        r = y - mu
        eta = family.link(mu)
        v = family.variance(mu)
        m_e = family.link.inverse_deriv(eta)
        weights = weights / weights.sum()

        rv = r / v * m_e * weights

        if scipy.sparse.issparse(X):
            xm, xs = design.xm, design.xs
            g = np.abs((X.T @ rv - np.sum(rv) * xm) / xs)
        else:
            g = np.abs(X.T @ rv)

        g = g * ju / (vp + (vp <= 0))
        lambda_max = np.max(g) / max(alpha, 1e-3)

        return {'nulldev':nulldev,
                'mu':mu,
                'lambda_max':lambda_max}

    def _get_initial_state(self,
                           warm=None):

        nobs, nvars = self.X.shape

        # get offset

        is_offset = self.offset is not None

        if not warm:
            start_val = self._get_start()
            nulldev = start_val['nulldev']
            mu = start_val['mu']
            fit = None
            coefold = np.zeros(nvars)   # initial coefs = 0
            eta = self.family.link(mu)
            intold = (eta - self.offset)[0]

        else:
            fit = warm
            if 'warm_fit' in warm:
                nulldev = fit['nulldev']
                coefold = fit.warm_fit.a   # prev value for coefficients
                intold = fit.warm_fit.aint    # prev value for intercept
            elif 'a0' in warm and 'beta' in warm:
                nulldev = self._get_start()['nulldev']

                coefold = warm['beta']   # prev value for coefficients
                intold = warm['a0']      # prev value for intercept
            else:
                raise ValueError("Invalid warm start object")

        state = GLMState(coef=coefold,
                         intercept=intold)
        state.update(self.design,
                     self.family,
                     self.offset)

        return state, nulldev

    def _fit(self,
             design,
             y,
             weights,
             state):

        elnet_est = ElNetEstimator(lambda_val=0,
                                   standardize=False,
                                   intercept=self.intercept)

        fit, converged, boundary, state = _IRLS(self,
                                                state,
                                                design,
                                                y,
                                                weights,
                                                elnet_est.fit)

        # checks on convergence and fitted values
        if not converged:
            warnings.warn("fitting IRLS: algorithm did not converge")
        if boundary:
            warnings.warn("fitting IRLS: algorithm stopped at boundary value")

        # create a GLMNetResult

        args = asdict(fit)
        args['offset'] = self.is_offset
        args['nulldev'] = self.nulldev

        args['dev_ratio'] = (1 - self.dev_function(state.mu) / self.nulldev)
        args['family'] = self.family
        args['converged'] = converged
        args['boundary'] = boundary
        args['obj_function'] = state.obj_val

        return GLMResult(**args)
add_dataclass_docstring(GLMEstimator, subs={'control':'control_glm'})

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

# private functions

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

    fit = elnet_solver(design, z, weights=w, warm=warm).result_

    if fit.jerr != 0:
        errmsg = _jerr_elnetfit(fit.jerr, spec.control.maxit)
        raise ValueError(errmsg['msg'])

    # update coefficients, eta, mu and obj_val
    # based on full quasi-newton step

    coefnew = fit.warm_fit.a
    intnew = fit.warm_fit.aint
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
def _quasi_newton_step(spec,
                       design,
                       y,
                       weights,
                       state,
                       elnet_solver,
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

    # do WLS with warmstart from previous iteration

    # IMPORTANT
    # passing `design` here ensures that the X will not be re-standardized
    # _get_design ignores "standardize" if X is a Design instance

    # WORKAROUND: use a cloned version of spec and set standardize=False

    fit = elnet_solver(design, z, weights=w).result_

    if fit.jerr != 0:
        errmsg = _jerr_elnetfit(fit.jerr, spec.control.maxit)
        raise ValueError(errmsg['msg'])

    # update coefficients, eta, mu and obj_val
    # based on full quasi-newton step

    coefnew = fit.warm_fit.a
    intnew = fit.warm_fit.aint
    state = GLMState(coefnew,
                     intnew,
                     obj_val_old=state.obj_val)

    state.update(design,
                 spec.family,
                 spec.offset)

    state.obj_val = _obj_function(y,
                                  state.mu,
                                  weights,
                                  spec.family,
                                  spec.lambda_val,
                                  spec.alpha,
                                  state.coef,
                                  spec.vp)

    # check to make sure it is a feasible descent step

    boundary = False
    halved = False  # did we have to halve the step size?

    # if objective function is not finite, keep halving the stepsize until it is finite
    # for the halving step, we probably have to adjust fit$g as well?

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
                state.obj_val = _obj_function(y,
                                              state.mu,
                                              weights,
                                              spec.family,
                                              spec.lambda_val,
                                              spec.alpha,
                                              state.coef,
                                              spec.vp)
                check, boundary_, halved_ = test(state)

    # if we did any halving, we have to update the coefficients, intercept
    # and weighted residual in the warm_fit object
    if halved:
        fit.warm_fit.a = state.coef
        fit.warm_fit.aint = state.intercept
        fit.warm_fit.r =  w * (z - state.eta)

    # test for convergence
    return state, fit, boundary, halved

def _IRLS(spec,
          design,
          y,
          weights,
          state,
          elnet_solver):

    # would be good to have objective and elnet_solver as args

    coefold, intold = state.coef, state.intercept

    state.obj_val_old = _obj_function(y,
                                      state.mu,
                                      weights,
                                      spec.family,
                                      spec.lambda_val,
                                      spec.alpha,
                                      state.coef,
                                      spec.vp)

    converged = False
    fit = None

    for iter in range(spec.control.mxitnr):

        (state,
         fit,
         boundary,
         halved) = _quasi_newton_step(spec,
                                      design,
                                      y,
                                      weights,
                                      state,
                                      elnet_solver,
                                      fit)

        print(state.obj_val, state.obj_val_old)

        # test for convergence
        if (np.fabs(state.obj_val - state.obj_val_old)/(0.1 + abs(state.obj_val)) < spec.control.epsnr):
            converged = True
            break

    return fit, converged, boundary, state
