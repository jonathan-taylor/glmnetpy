from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np

from sklearn.base import BaseEstimator

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .base import Design, _get_design, Penalty
from .docstrings import add_dataclass_docstring

from .elnet import (ElNetEstimator,
                    ElNetControl,
                    ElNetSpec)
from .glm import (GLMState,
                  _IRLS,
                  GLMEstimator)

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
    warm_fit: dict = field(default_factory=dict)
    nvars: InitVar[int] = None
    control: InitVar[GLMNetControl] = None
    
    def __post_init__(self, nvars, control):

        if self.lower_limits == -np.inf:
            self.lower_limits = -np.ones(nvars) * control.big
        
        if self.upper_limits == np.inf:
            self.upper_limits = np.ones(nvars) * control.big
        
        if self.penalty_factor is None:
            self.penalty_factor = np.ones(nvars)

        self.penalty_factor *= nvars / self.penalty_factor.sum() 
            
        self.elnet_estimator = ElNetEstimator(lambda_val=self.lambda_val,
                                              alpha=self.alpha,
                                              control=control,
                                              lower_limits=self.lower_limits,
                                              upper_limits=self.upper_limits,
                                              fit_intercept=self.fit_intercept,
                                              standardize=False)

    def quasi_newton_step(self,
                          design,
                          pseudo_response,
                          sample_weight):

        z = pseudo_response
        w = sample_weight

        # make sure to set lambda_val to self.lambda_val
        self.elnet_estimator = ElNetEstimator(lambda_val=self.lambda_val,
                                              alpha=self.alpha,
                                              control=self.elnet_estimator.control,
                                              lower_limits=self.lower_limits,
                                              upper_limits=self.upper_limits,
                                              fit_intercept=self.fit_intercept,
                                              standardize=False)
        
        out = self.elnet_estimator.fit(design, z, sample_weight=sample_weight).result_
        coefnew = out.beta.toarray().reshape(-1) # this will not have been scaled by `xs/scaling_`
        intnew = out.a0
        
        self.warm_fit['coef_'] = coefnew
        self.warm_fit['intercept_'] = intnew
        
        return coefnew, intnew

    def get_warm_start(self):

        if ('coef_' in self.warm_fit.keys() and
            'intercept_' in self.warm_fit.keys()):

            return GLMState(self.warm_fit['coef_'],
                            self.warm_fit['intercept_']) 

    def update_resid(self, r):
        self.warm_fit['resid_'] = r
        
    def objective(self, state):
        return 0
# end of GLMNetRegularizer

@dataclass
class GLMNetEstimator(GLMEstimator,
                      GLMNetSpec):

    control: GLMNetControl = field(default_factory=GLMNetControl)

    def _get_regularizer(self,
                         X):

        return GLMNetRegularizer(lambda_val=self.lambda_val,
                                 alpha=self.alpha,
                                 penalty_factor=self.penalty_factor,
                                 lower_limits=self.lower_limits,
                                 upper_limits=self.upper_limits,
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
            offset=None):
        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    exclude=exclude,
                    dispersion=1,
                    offset=offset)

        if self.standardize:
            self.scaling_ = self.design_.scaling_
            self.coef_ /= self.scaling_
        self.intercept_ -= (self.coef_ * self.design_.centers_).sum()

add_dataclass_docstring(GLMNetEstimator, subs={'control':'control_glm'})

