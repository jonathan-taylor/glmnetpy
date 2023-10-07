from typing import Union
from dataclasses import (dataclass,
                         asdict,
                         field)
from functools import partial
import logging
   
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import scipy.sparse
from scipy.stats import norm as normal_dbn
from scipy.stats import t as t_dbn

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from ._utils import (_parent_dataclass_from_child,
                     _get_data)

from .base import Design, _get_design
from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)
from .irls import IRLS

@add_dataclass_docstring
@dataclass
class GLMFamilySpec(object):
    
    base: sm_family.Family = field(default_factory=sm_family.Gaussian)

    def link(self,
             mu):
        return self.base.link(mu)
    
    def deviance(self,
                 y,
                 mu,
                 sample_weight):
        if sample_weight is not None:
            return self.base.deviance(y, mu, freq_weights=sample_weight)
        else:
            return self.base.deviance(y, mu)

    def null_fit(self,
                 y,
                 sample_weight,
                 fit_intercept):
        sample_weight = np.asarray(sample_weight)
        y = np.asarray(y)
        if fit_intercept:
            mu0 = (y * sample_weight).sum() * np.ones_like(y) / sample_weight.sum()
        else:
            mu0 = self.base.link.inverse(np.zeros(y.shape, float))
        return mu0

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
        intold = self.base.link(null_fit[0]) # a GLM family specific
        return GLMState(coef=coefold,
                        intercept=intold)

    def get_response_and_weights(self,
                                 state,
                                 y,
                                 offset,
                                 sample_weight):

        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        newton_weights = sample_weight * dmu_deta**2 / varmu

        pseudo_response = state.eta + (y - state.mu) / dmu_deta

        return pseudo_response, newton_weights
        
@add_dataclass_docstring
@dataclass
class GLMControl(object):

    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35
    logging: bool = False

@dataclass
class GLMBaseSpec(object):

    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    fit_intercept: bool = True
    control: GLMControl = field(default_factory=GLMControl)
    offset_col: Union[str,int] = None
    weight_col: Union[str,int] = None
    response_col: Union[str,int] = None

add_dataclass_docstring(GLMBaseSpec, subs={'control':'control_glm'})

@add_dataclass_docstring
@dataclass
class GLMResult(object):

    family: GLMFamilySpec
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float

@dataclass
class GLMState(object):

    coef: np.ndarray
    intercept: np.ndarray
    obj_val: float = np.inf
    pmin: float = 1e-9
    
    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset,
               objective=None):
        '''pin the mu/eta values to coef/intercept'''

        family = family.base
        self.eta = design @ self._stack
        if offset is None:
            self.linear_predictor = self.eta
        else:
            self.linear_predictor = self.eta + offset
        self.mu = family.link.inverse(self.linear_predictor)
        if isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.linear_predictor = family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):

        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.linear_predictor)
        
        # compute working residual
        r = (y - self.mu) 
        return sample_weight * r * dmu_deta / varmu 

@dataclass
class GLMRegularizer(object):

    fit_intercept: bool = False
    warm_state: dict = field(default_factory=dict)

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
                    sample_weight,
                    cur_state):   # ignored for GLM

        z = pseudo_response
        w = sample_weight

        if scipy.sparse.issparse(design.X):
            lm = LinearRegression(fit_intercept=self.fit_intercept)
            lm.fit(design.X, z, sample_weight=w)
            coefnew = lm.coef_
            intnew = lm.intercept_

        else:
            sqrt_w = np.sqrt(w)
            XW = design.X * sqrt_w[:, None]
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

    def objective(self, state):
        return 0

add_dataclass_docstring(GLMRegularizer, subs={'warm_state':'warm_state'})
# end of GLMRegularizer

@dataclass
class GLMBase(BaseEstimator,
              GLMBaseSpec):

    def _get_regularizer(self,
                         X):
        return GLMRegularizer(fit_intercept=self.fit_intercept)

    # no standardization for GLM
    def _get_design(self,
                    X,
                    sample_weight):
        return _get_design(X, sample_weight)

    def _get_family_spec(self,
                         y):
        if isinstance(self.family, sm_family.Family):
            return GLMFamilySpec(self.family)
        elif isinstance(self.family, GLMFamilySpec):
            return self.family

    def _check(self, X, y, check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_col=self.offset_col,
                         response_col=self.response_col,
                         weight_col=self.weight_col,
                         check=check)
    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            dispersion=1,
            check=True):

        nobs, nvar = X.shape
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)

        X, y, response, offset, weight = self._check(X, y, check=check)

        sample_weight = weight
        self.sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        response = np.asarray(response)

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        
        if not hasattr(self, "design_"):
            self.design_ = design = self._get_design(X, normed_sample_weight)
        else:
            design = self.design_
            
        (null_fit,
         self.null_deviance_) = self._family.get_null_deviance(response,
                                                               sample_weight,
                                                               self.fit_intercept)
        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(X)
        self.regularizer_ = regularizer

        if (hasattr(self.regularizer_, 'warm_state') and
            self.regularizer_.warm_state):
            state = self.regularizer_.warm_state
        else:
            state = self._family.get_null_state(null_fit,
                                                nvar)

        # for Cox, the state could have mu==eta so that this need not change
        def obj_function(response,
                         normed_sample_weight,
                         family,
                         regularizer,
                         state):
            val1 = family.deviance(response,
                                   state.mu,
                                   normed_sample_weight) / 2
            val2 = regularizer.objective(state)
            val = val1 + val2
            if self.control.logging: logging.debug(f'Computing objective, lambda: {regularizer.lambda_val}, alpha: {regularizer.alpha}, coef: {state.coef}, intercept: {state.intercept}, deviance: {val1}, penalty: {val2}')
            return val
        obj_function = partial(obj_function,
                               response.copy(),
                               normed_sample_weight.copy(),
                               self._family,
                               regularizer)
        
        state.update(design,
                     self._family,
                     offset,
                     obj_function)

        (converged,
         boundary,
         state,
         self._final_weights) = IRLS(regularizer,
                                     self._family,
                                     design,
                                     response,
                                     offset,
                                     normed_sample_weight,
                                     state,
                                     obj_function,
                                     self.control)

        # checks on convergence and fitted values
        if not converged:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm did not converge")
        if boundary:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm stopped at boundary value")

        self.deviance_ = self._family.deviance(response,
                                               state.mu,
                                               sample_weight) # not the normalized weights!

        self._set_coef_intercept(state)

        if (hasattr(self._family, "base") and 
            isinstance(self._family.base, sm_family.Gaussian)): # GLM specific
            # usual estimate of sigma^2
            self.dispersion_ = self.deviance_ / (nobs-nvar-self.fit_intercept) 
        else:
            self.dispersion_ = dispersion

        return self
    fit.__doc__ = '''
Fit a GLM.

Parameters
----------

{X}
{y}
{weights}
{exclude}
{summarize}
    
Returns
-------

self: object
        GLM class instance.
        '''.format(**_docstrings)
    
    def predict(self, X, prediction_type='response'):

        eta = X @ self.coef_ + self.intercept_
        if prediction_type == 'link':
            return eta
        elif prediction_type == 'response':
            return self._family.base.link.inverse(eta)
        else:
            raise ValueError("prediction should be one of 'response' or 'link'")
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

        mu = self.predict(X, prediction_type='response')
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        return -self._family.deviance(y, mu, sample_weight) / 2 # GLM specific
    score.__doc__ = '''
Compute weighted log-likelihood (i.e. negative deviance / 2) for test X and y using fitted model. Weights
default to `np.ones_like(y) / y.shape[0]`.

Parameters
----------

{X}
{y}
{sample_weight}    

Returns
-------

score: float
    Deviance of family for `(X, y, sample_weight)`.
'''.format(**_docstrings).strip()

    def _set_coef_intercept(self, state):
        self.coef_ = state.coef.copy() # this makes a copy -- important to make this copy because `state.coef` is persistent
        if hasattr(self, 'standardize') and self.standardize:
            self.scaling_ = self.design_.scaling_
            self.coef_ /= self.scaling_ 
        self.intercept_ = state.intercept - (self.coef_ * self.design_.centers_).sum()

GLMBase.__doc__ = '''
Base class to fit a Generalized Linear Model (GLM). Base class for `GLMNet`.

Parameters
----------
{fit_intercept}
{summarize}
{family}
{control_glm}

Attributes
__________
{coef_}
{intercept_}
{summary_}
{covariance_}
{null_deviance_}
{deviance_}
{dispersion_}
{regularizer_}
'''.format(**_docstrings)

@dataclass
class GLM(GLMBase):

    summarize: bool = False

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            dispersion=1,
            check=True):

        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    exclude=exclude,
                    dispersion=dispersion,
                    check=check)

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])
            
        if self.summarize:
            self.covariance_, self.summary_ = self._summarize(exclude,
                                                              self.dispersion_,
                                                              sample_weight, # not normalized!
                                                              X.shape)


        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   X_shape):

        # IRLS used normalized weights,
        # this unnormalizes them...
        unscaled_precision_ = self.design_.quadratic_form(self._final_weights * sample_weight.sum()) 

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

        family = self._family.base
        if (isinstance(family, sm_family.Gaussian) and
            isinstance(family.link, sm_links.Identity)):
            n, p = X_shape
            self.resid_df_ = n - p - self.fit_intercept
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     't': T,
                                     'P>|t|': 2 * t_dbn.sf(np.fabs(T), df=self.resid_df_)},
                                    index=index)
        else:
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     'z': T,
                                     'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                    index=index)
        return covariance_, summary_
        
    
@dataclass
class GaussianGLM(RegressorMixin, GLM):

    def __post_init__(self):

        if (not hasattr(self._family, 'base')
            or not isinstance(self._family.base, sm_family.Gaussian)):
            msg = f'{self.__class__.__name__} expects a Gaussian family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)

@dataclass
class BinomialGLM(ClassifierMixin, GLM):

    family: sm_family.Family = field(default_factory=sm_family.Binomial)

    def __post_init__(self):

        if (not hasattr(self._family, 'base')
            or not isinstance(self.family.base, sm_family.Binomial)):
            msg = f'{self.__class__.__name__} expects a Binomial family.'
            warnings.warn(msg)
            if self.control.logging: logging.warn(msg)

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            dispersion=1,
            check=True):

        label_encoder = LabelEncoder().fit(y)
        if len(label_encoder.classes_) > 2:
            raise ValueError("BinomialGLM expecting a binary classification problem.")
        self.classes_ = label_encoder.classes_

        y_binary = label_encoder.transform(y)

        return super().fit(X,
                           y_binary,
                           sample_weight=sample_weight,
                           regularizer=regularizer,             # last 4 options non sklearn API
                           exclude=exclude,
                           dispersion=dispersion,
                           check=check)

    def predict(self, X, prediction_type='class'):

        eta = X @ self.coef_ + self.intercept_
        family = self._family.base
        if prediction_type == 'link':
            return eta
        elif prediction_type == 'response':
            return family.link.inverse(eta)
        elif prediction_type == 'class':
            pi_hat = family.link.inverse(eta)
            _integer_classes = (pi_hat > 0.5).astype(int)
            return self.classes_[_integer_classes]
        else:
            raise ValueError("prediction should be one of 'response', 'link' or 'class'")
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
