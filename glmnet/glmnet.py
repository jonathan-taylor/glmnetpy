from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np

from sklearn.base import BaseEstimator

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .base import _get_design, Penalty
from .docstrings import add_dataclass_docstring

from .elnet import (ElNet,
                    ElNetControl,
                    ElNetSpec)
from .glm import (GLMState,
                  GLM)

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
class GLMNetRegularizer(Penalty):

    fit_intercept: bool = False
    warm_state: dict = field(default_factory=dict)
    nvars: InitVar[int] = None
    control: InitVar[GLMNetControl] = None

    def __post_init__(self, nvars, control):

        self.lower_limits = np.asarray(self.lower_limits)
        if self.lower_limits.shape == (): # a single float 
            self.lower_limits = np.ones(nvars) * self.lower_limits
        
        self.upper_limits = np.asarray(self.upper_limits)
        if self.upper_limits.shape == (): # a single float 
            self.upper_limits = np.ones(nvars) * self.upper_limits
        
        if self.penalty_factor is None:
            self.penalty_factor = np.ones(nvars)

        self.penalty_factor *= nvars / self.penalty_factor.sum() 
            
        self.elnet_estimator = ElNet(lambda_val=self.lambda_val,
                                     alpha=self.alpha,
                                     control=control,
                                     lower_limits=self.lower_limits,
                                     upper_limits=self.upper_limits,
                                     fit_intercept=self.fit_intercept,
                                     penalty_factor=self.penalty_factor,
                                     standardize=False)

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
                    normed_sample_weight,
                    cur_state):
                            
        z = pseudo_response
        # make sure to set lambda_val to self.lambda_val
        self.elnet_estimator.lambda_val = self.lambda_val
        
        warm = (cur_state.coef,
                cur_state.intercept,
                cur_state.linear_predictor) # linear_predictor includes offet if any

        elnet_fit = self.elnet_estimator.fit(design,
                                             z,
                                             sample_weight=normed_sample_weight,
                                             warm=warm,
                                             check=False)
        
        self.warm_state = GLMState(elnet_fit.raw_coef_,
                                   elnet_fit.raw_intercept_)

        return self.warm_state

    def objective(self, state):
        lasso = self.alpha * np.fabs(state.coef).sum()
        ridge = (1 - self.alpha) * (state.coef**2).sum() / 2
        return self.lambda_val * (lasso + ridge)


# end of GLMNetRegularizer

@dataclass
class GLMNet(GLM,
             GLMNetSpec):

    control: GLMNetControl = field(default_factory=GLMNetControl)

    def _get_regularizer(self,
                         X):

        # self.design_ will have been set by now

        return GLMNetRegularizer(lambda_val=self.lambda_val,
                                 alpha=self.alpha,
                                 penalty_factor=self.penalty_factor,
                                 lower_limits=self.lower_limits * self.design_.scaling_,
                                 upper_limits=self.upper_limits * self.design_.scaling_,
                                 fit_intercept=self.fit_intercept,
                                 nvars=X.shape[1],
                                 control=self.control)

    # no standardization for GLM
    def _get_design(self,
                    X,
                    sample_weight):
        return _get_design(X,
                           sample_weight,
                           standardize=self.standardize,
                           intercept=self.fit_intercept)

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            offset=None,
            check=True):

        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    exclude=exclude,
                    dispersion=1,
                    offset=offset,
                    check=check)

        return self

add_dataclass_docstring(GLMNet, subs={'control':'control_glm'})

