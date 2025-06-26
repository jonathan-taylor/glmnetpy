from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)

from .base import _get_design, Penalty
from .docstrings import (add_dataclass_docstring,
                         _docstrings)

from .elnet import (ElNet,
                    ElNetControl,
                    ElNetSpec)
from .glm import (GLMFamilySpec,
                  GLM)

@add_dataclass_docstring
@dataclass
class RegGLMControl(ElNetControl):

    thresh: float = 1e-10
    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35
    logging: bool = False

@dataclass
class RegGLMSpec(ElNetSpec):

    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    control: RegGLMControl = field(default_factory=RegGLMControl)

add_dataclass_docstring(RegGLMSpec, subs={'control':'control_glmnet'})

@add_dataclass_docstring
@dataclass
class RegGLMResult(object):

    family: GLMFamilySpec
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float

@dataclass
class ElNetRegularizer(Penalty):

    fit_intercept: bool = False
    warm_state: dict = field(default_factory=dict)
    nvars: InitVar[int] = None
    control: InitVar[RegGLMControl] = None
    exclude: list = field(default_factory=list)

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
                                     standardize=False,
                                     exclude=self.exclude)

    def half_step(self,
                  state,
                  oldstate):
        klass = oldstate.__class__
        return klass(0.5 * (oldstate.coef + state.coef),
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
                cur_state.linear_predictor) # just X\beta -- doesn't include offset

        elnet_fit = self.elnet_estimator.fit(design,
                                             z,
                                             sample_weight=normed_sample_weight,
                                             warm=warm,
                                             check=False)
        
        klass = cur_state.__class__
        self.warm_state = klass(elnet_fit.raw_coef_,
                                elnet_fit.raw_intercept_)

        return self.warm_state

    def objective(self, state):
        lasso = self.alpha * np.fabs(state.coef).sum()
        ridge = (1 - self.alpha) * (state.coef**2).sum() / 2
        return self.lambda_val * (lasso + ridge)


# end of ElNetRegularizer

@dataclass
class RegGLM(GLM,
             RegGLMSpec):

    control: RegGLMControl = field(default_factory=RegGLMControl)

    def get_LM(self):
        return GLM(family=self.family,
                   fit_intercept=self.fit_intercept,
                   offset_id=self.offset_id,
                   weight_id=self.weight_id,
                   response_id=self.response_id)

    def _get_regularizer(self,
                         nvars=None):

        # self.design_ will have been set by now

        if nvars is None:
            check_is_fitted(self, ["design_"])
            nvars = self.design_.X.shape[1]

        return ElNetRegularizer(lambda_val=self.lambda_val,
                                alpha=self.alpha,
                                penalty_factor=self.penalty_factor,
                                lower_limits=self.lower_limits * self.design_.scaling_,
                                upper_limits=self.upper_limits * self.design_.scaling_,
                                fit_intercept=self.fit_intercept,
                                nvars=nvars,
                                control=self.control,
                                exclude=self.exclude)

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
            sample_weight=None,           # ignored
            regularizer=None,             # last 3 options non sklearn API
            warm_state=None,
            check=True,
            fit_null=True):

        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=1,
                    check=check)

        return self

add_dataclass_docstring(RegGLM, subs={'control':'control_glm'})

@dataclass
class GaussianRegGLM(RegressorMixin, RegGLM):

    def __post_init__(self):

        if self._family.is_gaussian:
            msg = f'{self.__class__.__name__} expects a Gaussian family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)

@dataclass
class BinomialRegGLM(ClassifierMixin, RegGLM):

    family: BinomFamilySpec = field(default_factory=BinomFamilySpec)

    def __post_init__(self):

        if not self._family.is_binomial:
            msg = f'{self.__class__.__name__} expects a Binomial family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):

        label_encoder = LabelEncoder().fit(y)
        if len(label_encoder.classes_) > 2:
            raise ValueError("BinomialRegGLM expecting a binary classification problem.")
        self.classes_ = label_encoder.classes_

        y_binary = label_encoder.transform(y)

        return super().fit(X,
                           y_binary,
                           sample_weight=sample_weight,
                           regularizer=regularizer,             # last 4 options non sklearn API
                           dispersion=dispersion,
                           offset=offset,
                           check=check)

    def predict(self, X, prediction_type='class'):

        if not hasattr(self.family, 'base'):
            raise ValueError(f'{self.__class__} expects to have a base family')
        linpred = X @ self.coef_ + self.intercept_ # often called eta
        pred = self._family.predict(linpred,
                                    prediction_type=prediction_type)
        if prediction_type == 'class':
            pred = self._classes[pred]

    predict.__doc__ = '''
Predict outcome of corresponding family.

Parameters
----------

{X}
{prediction_type_binomial}

Returns
-------

{prediction}'''.format(**_docstrings).strip()

    def predict_proba(self, X):

        '''

        Probability estimates for a BinomialGLM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
 
        '''

        prob_1 = self.predict(X, prediction_type='response')
        result = np.empty((prob_1.shape[0], 2))
        result[:,1] = prob_1
        result[:,0] = 1 - prob_1

        return result
