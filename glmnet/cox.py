from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from functools import partial

import numpy as np
import pandas as pd

from scipy.stats import norm as normal_dbn

from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

from coxdev import CoxDeviance
    
from .glm import (GLMFamilySpec,
                  GLMState,
                  GLM)
from .scoring import Scorer
from .regularized_glm import RegGLM
from .glmnet import GLMNet
from ._utils import _get_data

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

        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = self.link_parameter
        
        # shorthand
        self.mu = self.link_parameter
        self.eta = self.linear_predictor
        
        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):
        link_parameter = self.link_parameter
        family._result = family._coxdev(link_parameter,
                                        sample_weight)
        # the gradient is the gradient of the deviance
        # we want deviance of the log-likelihood
        return - family._result.gradient / 2

@dataclass
class CoxFamily(object):

    tie_breaking: Literal['breslow', 'efron'] = 'efron'
    event_id: Optional[str] = 'event'
    status_id: Optional[str] = 'status'
    start_id: Optional[str] = None

@dataclass
class CoxFamilySpec(object):

    event_data: InitVar[np.ndarray]
    tie_breaking: Literal['breslow', 'efron'] = 'efron'
    event_id: Optional[str] = 'event'
    status_id: Optional[str] = 'status'
    start_id: Optional[str] = None
    name: str = 'Cox'
    
    def __hash__(self):

        return (self.tie_breaking,
                self.event_id,
                self.status_id,
                self.start_id,
                self.name).__hash__()

    def __post_init__(self, event_data):

        if (self.event_id not in event_data.columns or
            self.status_id not in event_data.columns):
            raise ValueError(f'expecting f{event_id} and f{status_id} columns')
        
        event = event_data[self.event_id]
        status = event_data[self.status_id]
        
        if self.start_id is not None:
            start = event_data[self.start_id]
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

        link_parameter = mu
        self._result = self._coxdev(link_parameter,
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
                          offset, # ignored for Cox
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

        link_parameter = state.link_parameter
        linear_predictor = state.linear_predictor
        self._result = self._coxdev(link_parameter,
                                    sample_weight)
        # self._coxdev computes value, gradient and hessian of deviance
        # we want the gradient, hessian of deviance / 2
        gradient = self._result.gradient / 2
        diag_hessian = self._result.diag_hessian / 2
        test = diag_hessian != 0
        newton_weights = diag_hessian
        inv_weights = np.where(test, 1 / (diag_hessian + (1 - test)), 0)
        pseudo_response = linear_predictor - gradient * inv_weights

        return pseudo_response, newton_weights
    
    def information(self,
                    state,
                    sample_weight):
        return self._coxdev.information(state.link_parameter,
                                        sample_weight)

    def _default_scorers(self):

        return [CoxDiffScorer(coxfam=self), CoxScorer(coxfam=self)]

@dataclass
class CoxLM(GLM):
    
    fit_intercept: Literal[False] = False

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id)

    def _check(self,
               X,
               y,
               check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   X_shape):

        # IRLS used normalized weights,
        # this unnormalizes them...

        unscaled_precision_ = self.design_.quadratic_form(self._information)
        
        keep = np.ones(unscaled_precision_.shape[0]-1, bool)
        if exclude is not []:
            keep[exclude] = 0
        keep = np.hstack([self.fit_intercept, keep]).astype(bool)
        covariance_ = dispersion * np.linalg.inv(unscaled_precision_[keep][:,keep])

        SE = np.sqrt(np.diag(covariance_)) 
        index = self.feature_names_in_
        if self.fit_intercept:
            coef = np.hstack([self.intercept_, self.coef_])
            T = np.hstack([self.intercept_ / SE[0], self.coef_ / SE[1:]])
            index = ['intercept'] + index
        else:
            coef = self.coef_
            T = self.coef_ / SE

        summary_ = pd.DataFrame({'coef':coef,
                                 'std err': SE,
                                 'z': T,
                                 'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                index=index)
        return covariance_, summary_


@dataclass
class RegCoxLM(RegGLM):
    
    fit_intercept: Literal[False] = False

    def _check(self, X, y, check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id)

@dataclass
class CoxNet(GLMNet):
    
    fit_intercept: Literal[False] = False
    regularized_estimator: BaseEstimator = RegCoxLM
    
    def _check(self,
               X,
               y,
               check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _get_family_spec(self,
                         y):
        event_data = y
        return CoxFamilySpec(event_data=event_data,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id)
    
    def get_LM(self):
        return CoxLM(family=self.family,
                     offset_id=self.offset_id,
                     weight_id=self.weight_id,
                     response_id=self.response_id)

    def _get_initial_state(self,
                           X,
                           y,
                           exclude):

        n, p = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor == 0
        keep[exclude] = 0

        coef_ = np.zeros(p)
        intercept_ = 0

        if keep.sum() > 0:
            X_keep = X[:,keep]

            coxlm = self.get_LM()
            coxlm.fit(X_keep, y)
            coef_[keep] = coxlm.coef_

        return CoxState(coef_, intercept_), keep.astype(float)

    def predict(self,
                X):
        
        linear_pred_ = self.coefs_ @ X.T + self.intercepts_[:, None]
        linear_pred_ = linear_pred_.T

        # make return based on original
        # promised number of lambdas
        # pad with last value
        if self.lambda_values is not None:
            nlambda = self.lambda_values.shape[0]
        else:
            nlambda = self.nlambda

        value = np.zeros((linear_pred_.shape[0], nlambda), float) * np.nan
        value[:,:linear_pred_.shape[1]] = linear_pred_
        value[:,linear_pred_.shape[1]:] = linear_pred_[:,-1][:,None]
        return value

    
@dataclass(frozen=True)
class CoxScorer(Scorer):

    coxfam: CoxFamilySpec=None
    use_full_data: bool=True
    maximize: bool=False

    name: str = 'Cox Deviance'

    def score_fn(self,
                 split,
                 event_data,
                 predictions,
                 sample_weight):
        
        coxfam = self.coxfam
        sample_weight = np.asarray(sample_weight)
        status = np.asarray(event_data[coxfam.status_id])

        cls = self.coxfam.__class__

        predictions = np.asarray(predictions)
        event_split = event_data.iloc[split]
        fam_split = cls(tie_breaking=coxfam.tie_breaking,
                        event_id=coxfam.event_id,
                        status_id=coxfam.status_id,
                        start_id=coxfam.start_id,
                        event_data=event_split)

        split_w = sample_weight[split]
        dev_split = fam_split._coxdev(predictions[split], split_w).deviance
        w_sum = sample_weight[split].sum()

        return dev_split / w_sum, w_sum

@dataclass(frozen=True)
class CoxDiffScorer(CoxScorer):

    name: str = 'Cox Deviance (Difference)'
    def score_fn(self,
                 split,
                 event_data,
                 predictions,
                 sample_weight):
        
        coxfam = self.coxfam
        sample_weight = np.asarray(sample_weight)
        status = np.asarray(event_data[coxfam.status_id])
        cls = self.coxfam.__class__
        predictions = np.asarray(predictions)

        fam_full = cls(tie_breaking=coxfam.tie_breaking,
                       event_id=coxfam.event_id,
                       status_id=coxfam.status_id,
                       start_id=coxfam.start_id,
                       event_data=event_data)
        dev_full = fam_full._coxdev(predictions, sample_weight).deviance

        # now compute deviance on complement
        
        split_c = np.ones_like(predictions, bool)
        split_c[split] = 0

        event_c = event_data.iloc[split_c] # XXX presumes dataframe, could be ndarray
        fam_split_c = cls(tie_breaking=coxfam.tie_breaking,
                          event_id=coxfam.event_id,
                          status_id=coxfam.status_id,
                          start_id=coxfam.start_id,
                          event_data=event_c)
        split_c_w = sample_weight[split_c]
        dev_c = fam_split_c._coxdev(predictions[split_c], split_c_w).deviance

        w_sum = sample_weight.sum() - split_c_w.sum()
        return (dev_full - dev_c) / w_sum, w_sum


    




  
