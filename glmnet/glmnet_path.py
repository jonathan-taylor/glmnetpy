from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np

from sklearn.base import BaseEstimator

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .base import Design, _get_design, Penalty
from .docstrings import add_dataclass_docstring

from .glmnet import (GLMNetControl,
                     GLMNet)
from .glm import GLM, GLMState

@dataclass
class GLMNetPathSpec(object):

    lambda_values : np.ndarray
    lambda_fractional: bool = True
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None
    fit_intercept: bool = True
    standardize: bool = True
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMNetControl = field(default_factory=GLMNetControl)

add_dataclass_docstring(GLMNetPathSpec, subs={'control':'control_glmnet'})

@dataclass
class GLMNetPath(BaseEstimator,
                 GLMNetPathSpec):

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            offset=None):

        self.glmnet_est_ = GLMNet(lambda_val=self.control.big,
                                  family=self.family,
                                  alpha=self.alpha,
                                  penalty_factor=self.penalty_factor,
                                  lower_limits=self.lower_limits,
                                  upper_limits=self.upper_limits,
                                  fit_intercept=self.fit_intercept,
                                  control=self.control)
        self.glmnet_est_.fit(X, y, sample_weight)
        regularizer_ = self.glmnet_est_.regularizer_

        state, keep_ = self._get_initial_state(X,
                                               y,
                                               sample_weight,
                                               exclude,
                                               offset)
        state.update(self.glmnet_est_.design_,
                     self.family,
                     offset)

        logl_score = state.logl_score(self.family,
                                      y)
        score_ = (self.glmnet_est_.design_.T @ (sample_weight * logl_score))[1:]
        pf = regularizer_.penalty_factor
        score_ /= (pf + (pf ==0))
        score_[exclude] = 0
        self.lambda_max_ = np.fabs(score_).max()

        if self.lambda_fractional:
            self.lambda_values_ = np.sort(self.lambda_max_ * self.lambda_values)[::-1]
        else:
            self.lambda_values_ = np.sort(self.lambda_values)[::-1]

        coefs_ = []
        intercepts_ = []
        for l in self.lambda_values_:

            self.glmnet_est_.lambda_val = regularizer_.lambda_val = l
            self.glmnet_est_.fit(X,
                                 y,
                                 sample_weight,
                                 offset=offset,
                                 regularizer=regularizer_)
            coefs_.append(self.glmnet_est_.coef_.copy())
            intercepts_.append(self.glmnet_est_.intercept_)

        self.coefs_ = np.array(coefs_)
        self.intercepts_ = np.array(intercepts_)

        return self
    
    def _get_initial_state(self,
                           X,
                           y,
                           sample_weight,
                           exclude,
                           offset):

        n, p = X.shape
        keep = self.glmnet_est_.regularizer_.penalty_factor == 0
        keep[exclude] = 0

        coef_ = np.zeros(p)

        if keep.sum() > 0:
            X_keep = X[:,keep]

            glm = GLM(fit_intercept=self.fit_intercept,
                      family=self.family)
            glm.fit(X_keep, y, sample_weight, offset=offset)
            coef_[keep] = glm.coef_
            intercept_ = glm.intercept_
        else:
            intercept_ = self.family.link(y.mean())
        return GLMState(coef_, intercept_), keep.astype(float)

