from dataclasses import dataclass
import logging

import numpy as np
import scipy.sparse
from copy import deepcopy

# for Gaussian check below
from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
from .glm import GLMRegularizer
from .base import _get_design

def quasi_newton_step(regularizer,
                      family,
                      design,
                      y,
                      offset,
                      weights,
                      state,
                      objective,
                      control,
                      step_search=False):

    oldstate = deepcopy(state)
    
    pseudo_response, newton_weights = family.get_response_and_weights(state,
                                                                      y,
                                                                      offset,
                                                                      weights)

    state = regularizer.newton_step(design,
                                    pseudo_response,
                                    newton_weights,
                                    state)

    state.update(design,
                 family,
                 offset,
                 objective)

    half_step = regularizer.half_step(oldstate,
                                      state)

    half_step.update(design,
                     family,
                     offset,
                     objective)
    boundary = False
    halved = False 

    return half_step, boundary, halved, newton_weights

def GLMM(family,
         fixed_effect_design,
         latent_design,
         y,
         weights,
         control,
         precision,
         seed=345):

    rng = np.random.default_rng(seed)

    sigmasq_state = 1.

    fixed_effect_regularizer = GLMRegularizer(fit_intercept=True)
    latent_regularizer = RidgeRegularizer(precision=precision, scale=np.sqrt(sigmasq_state))

    null_fit, _ = family.get_null_deviance(y,
                                           weights,
                                           True)

    # TODO? update the null state in get_null_state?
    fixed_effect_state = family.get_null_state(null_fit,
                                               fixed_effect_design.X.shape[1])
    fixed_effect_state.update(fixed_effect_design,
                              family,
                              np.zeros_like(y),
                              None)

    latent_state = family.get_null_state(null_fit,
                                         latent_design.X.shape[1])
    latent_state.update(latent_design,
                        family,
                        np.zeros_like(y),
                        None)

    if control.logging:
        logging.info('Starting GLMM')

    history = []
    sigmasq_history = []
    
    for i in range(control.mxitnr):

        # take a step of fixed effect
        fixed_effect_offset = latent_design @ latent_state._stack

        (fixed_effect_state,
         _,
         _,
         _) = quasi_newton_step(fixed_effect_regularizer,
                                family,
                                fixed_effect_design,
                                y,
                                fixed_effect_offset,
                                weights,
                                fixed_effect_state,
                                None,
                                control)

        # now latent state
        latent_offset = fixed_effect_design @ fixed_effect_state._stack
        (latent_state,
         _,
         _,
         _) = quasi_newton_step(latent_regularizer,
                                family,
                                latent_design,
                                y,
                                latent_offset,
                                weights,
                                latent_state,
                                None,
                                control)
        history.append(fixed_effect_state._stack)

        # update sigma^2

#        sigmasq_state = 2
        q = latent_state.coef.shape[0]
        sigmasq_hat = np.linalg.norm(latent_state.coef)**2 / q
        sigmasq_cur = 0.5 * sigmasq_hat + 0.5 * sigmasq_state
        sigmasq_state = np.clip(sigmasq_cur + rng.standard_normal() * sigmasq_state * np.sqrt(2 / q), 1e-5, 2)

#        print(sigmasq_state)
        latent_regularizer.scale = np.sqrt(sigmasq_state)
        sigmasq_history.append(sigmasq_state)
        
    return np.asarray(history), np.asarray(sigmasq_history)

@dataclass
class RidgeRegularizer(GLMRegularizer):

    precision: np.ndarray = None
    seed: int = 666
    scale: float = 1.
    
    def __post_init__(self):

        self._sqrt_prec = np.linalg.cholesky(self.precision)
        self._rng = np.random.default_rng(self.seed)

    def sample(self):

        # NOTE: check the math -- don't think we need to invert precision here...
        _q = self._sqrt_prec.shape[0]
        return self._sqrt_prec @ self._rng.standard_normal(_q) / self.scale
    
    def newton_step(self,
                    design,
                    pseudo_response,
                    sample_weight,
                    cur_state):   # ignored for GLM

        _q = self._sqrt_prec.shape[0]

        z = np.hstack([pseudo_response, self.sample()])
        w = np.hstack([sample_weight, np.ones(_q)])
        
        if scipy.sparse.issparse(design.X):
            raise NotImplementedError
            lm = LinearRegression(fit_intercept=self.fit_intercept)
            lm.fit(design.X, z, sample_weight=w)
            coefnew = lm.coef_
            intnew = lm.intercept_

        else:
            sqrt_w = np.sqrt(w)
            XW = np.concatenate([design.X,
                                 self._sqrt_prec.T / self.scale], axis=0) * sqrt_w[:, None]
            if self.fit_intercept:
                Wz = sqrt_w * z
                XW = np.concatenate([sqrt_w.reshape((-1,1)), XW], axis=1)
                Q = XW.T @ XW
                V = XW.T @ Wz
                try:
                    beta = np.linalg.solve(Q, V)
                except LinAlgError as e:
                    if self.control.logging: logging.debug("Error in solve: possible singular matrix, trying pseudo-inverse")
                    beta = np.linalg.pinv(XW) @ Wz
                coefnew = beta[1:]
                intnew = beta[0]

            else:
                coefnew = np.linalg.pinv(XW) @ (sqrt_w * z)
                intnew = 0

        klass = cur_state.__class__
        self.warm_state = klass(coefnew,
                                intnew)
        
        return self.warm_state

