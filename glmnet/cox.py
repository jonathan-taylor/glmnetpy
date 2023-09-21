from dataclasses import dataclass, InitVar
from typing import Optional, Literal

import numpy as np
import pandas as pd
from joblib import hash

from sklearn.utils import check_X_y

from coxdev import CoxDeviance

@dataclass
class CoxDevianceResult(object):

    loglik_sat: float
    deviance: float
    gradient: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]
    hash: str
    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)

@dataclass
class CoxState(GLMState):

    coef: np.ndarray
    obj_val: float = np.inf
    intercept: float = 0

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
        self.mu = self.linear_predictor

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y):

        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.linear_predictor)
        
        # compute working residual
        r = (y - self.mu) 
        return r * dmu_deta / varmu 

@dataclass
class CoxFamily(object):

    tie_breaking: Literal['breslow', 'efron'] = 'efron'
    event_col: Optional[str] = 'event'
    status_col: Optional[str] = 'status'
    start_col: Optional[str] = None

@dataclass
class CoxFamilySpec(object):

    event_data: InitVar[np.ndarray]
    tie_breaking: Literal['breslow', 'efron'] = 'efron'
    event_col: Optional[str] = 'event'
    status_col: Optional[str] = 'status'
    start_col: Optional[str] = None

    def __post_init__(self, event_data):

        if (self.event_col not in event_data.columns or
            self.status_col not in event_data.columns):
            raise ValueError(f'expecting f{event_col} and f{status_col} columns')
        
        event = event_data[self.event_col]
        status = event_data[self.status_col]
        
        if self.start_col is not None:
            start = event_data[self.start_col]
        else:
            start = None

        self._coxdev = CoxDeviance(event,
                                   status,
                                   start=start,
                                   tie_breaking=self.tie_breaking)

    # GLMFamilySpec API
    def link(self, mu):
        return mu
    
    def deviance(self, 
                 y,
                 mu,
                 sample_weight):

        linear_predictor = mu
        self._result = CoxDevianceResult(*self._coxdev(linear_predictor,
                                                       sample_weight),
                                         hash=hash([linear_predictor,
                                                    sample_weight]))
        return self._result.deviance
    
    def null_fit(self,
                 y,
                 sample_weight,
                 fit_intercept):
        sample_weight = np.asarray(sample_weight)
        return np.zeros_like(sample_weight)

    def get_null_deviance(self,
                          y,
                          sample_weight,
                          fit_intercept):
        mu0 = self.null_fit(y, sample_weight, fit_intercept)
        return mu0, self.deviance(y, mu0, sample_weight)

    def get_null_state(self,
                       null_fit,
                       nvars):
        coefold = np.zeros(nvars)   # initial coefs = 0
        return CoxState(coef=coefold,
                        intercept=0)

    def get_response_and_weights(self,
                                 state,
                                 y,
                                 offset,
                                 sample_weight):

        linear_predictor = state.linear_predictor
        _hash = hash([linear_predictor, sample_weight])
        if not hasattr(self, "_result") or self._result.hash != _hash:
            self._result = CoxDevianceResult(*self._coxdev(linear_predictor,
                                                           sample_weight),
                                             hash=hash([linear_predictor,
                                                        sample_weight]))
        else:
            print('got it')
        gradient = self._result.gradient
        diag_hessian = self._result.diag_hessian
        test = diag_hessian != 0
        newton_weights = np.where(test, sample_weight / (diag_hessian + (1 - test)), 0)
        inv_weights = np.where(test, 1 / (diag_hessian + (1 - test)), 0)
        if offset is not None:
            pseudo_response = (linear_predictor - offset) - gradient * inv_weights
        else:
            pseudo_response = linear_predictor - gradient * inv_weights

        return pseudo_response, newton_weights
    

@dataclass
class CoxLM(GLM):
    
    fit_intercept: bool = False

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_col=self.family.event_col,
                             status_col=self.family.status_col,
                             start_col=self.family.start_col)

    def _check(self, X, y):
        return check_X_y(X, y,
                         accept_sparse=['csc'],
                         multi_output=True,
                         estimator=self)



  
