from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from functools import partial

import numpy as np
import pandas as pd

from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

from coxdev import CoxDeviance
    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)
from .regularized_glm import RegGLM
from .glmnet import GLMNet

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
                   y,
                   sample_weight):
        linear_predictor = self.linear_predictor
        family._result = family._coxdev(linear_predictor,
                                        sample_weight)
        # the gradient is the gradient of the deviance
        # we want deviance of the log-likelihood
        return - family._result.gradient / 2

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
    name: str = 'Cox'
    
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
        self._result = self._coxdev(linear_predictor,
                                    sample_weight)
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
        self._result = self._coxdev(linear_predictor,
                                    sample_weight)
        # self._coxdev computes value, gradient and hessian of deviance
        # we want the gradient, hessian of deviance / 2
        gradient = self._result.gradient / 2
        diag_hessian = self._result.diag_hessian / 2
        test = diag_hessian != 0
        newton_weights = diag_hessian
        inv_weights = np.where(test, 1 / (diag_hessian + (1 - test)), 0)
        if offset is not None:
            pseudo_response = (linear_predictor - offset) - gradient * inv_weights
        else:
            pseudo_response = linear_predictor - gradient * inv_weights

        return pseudo_response, newton_weights
    

@dataclass
class CoxLM(GLM):
    
    fit_intercept: Literal[False] = False

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

@dataclass
class RegCoxLM(RegGLM):
    
    fit_intercept: Literal[False] = False

    def _check(self, X, y):
        return check_X_y(X, y,
                         accept_sparse=['csc'],
                         multi_output=True,
                         estimator=self)

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_col=self.family.event_col,
                             status_col=self.family.status_col,
                             start_col=self.family.start_col)

@dataclass
class CoxNet(GLMNet):
    
    fit_intercept: Literal[False] = False
    regularized_estimator: BaseEstimator = RegCoxLM
    
    def _check(self, X, y):
        X, _ =  check_X_y(X, y,
                          accept_sparse=['csc'],
                          multi_output=True,
                          estimator=self)
        return X, y

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_col=self.family.event_col,
                             status_col=self.family.status_col,
                             start_col=self.family.start_col)
    
    def _get_initial_state(self,
                           X,
                           y,
                           sample_weight,
                           exclude,
                           offset):

        n, p = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor == 0
        keep[exclude] = 0

        coef_ = np.zeros(p)
        intercept_ = 0

        if keep.sum() > 0:
            X_keep = X[:,keep]

            coxlm = CoxLM(family=self.family)
            coxlm.fit(X_keep, y, sample_weight, offset=offset)
            coef_[keep] = coxlm.coef_

        return CoxState(coef_, intercept_), keep.astype(float)

    def predict(self,
                X):
        
        linear_pred_ = self.coefs_ @ X.T + self.intercepts_[:, None]
        linear_pred_ = linear_pred_.T
        return linear_pred_
        
    def _get_scores(self,
                    y,
                    predictions,
                    test_splits,
                    scorers=[]):

        event_data = y

        scores_ = []

        if hasattr(self._family, 'base'):
            fam_name = self._family.base.__class__.__name__
        else:
            fam_name = self._family.__class__.__name__

        def _dev(family, event_data, eta, sample_weight):
            fam = CoxFamilySpec(tie_breaking=family.tie_breaking,
                                event_col=family.event_col,
                                status_col=family.status_col,
                                start_col=family.start_col,
                                event_data=event_data)
            return fam._coxdev(eta, sample_weight).deviance / event_data.shape[0]
        _dev = partial(_dev, self.family)

        if scorers is None:
            # create default scorers
            scorers_ = [(f'{self._family.name} Deviance', _dev, 'min')]

        else:
            scorers_ = scorers
            
        for split in test_splits:
            preds_ = predictions[split]
            y_ = event_data.iloc[split]
            w_ = np.ones(y_.shape[0])
            scores_.append([[score(y_, preds_[:,i], sample_weight=w_) for _, score, _ in scorers_]
                            for i in range(preds_.shape[1])])

        return scorers_, np.array(scores_)

    


  
