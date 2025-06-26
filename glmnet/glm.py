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
from scipy.sparse.linalg import LinearOperator
from scipy.stats import norm as normal_dbn
from scipy.stats import t as t_dbn

from sklearn.base import (BaseEstimator,
                          ClassifierMixin,
                          RegressorMixin)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import LabelEncoder

from ._utils import (_parent_dataclass_from_child,
                     _get_data)

from .base import (Design,
                   _get_design)

from .docstrings import (make_docstring,
                         add_dataclass_docstring,
                         _docstrings)
from .irls import IRLS
from .family import (GLMFamilySpec,
                     BinomFamilySpec,
                     GLMState)

@add_dataclass_docstring
@dataclass
class GLMControl(object):

    mxitnr: int = 25
    epsnr: float = 1e-6
    big: float = 9.9e35
    logging: bool = False

@dataclass
class GLMBaseSpec(object):

    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    fit_intercept: bool = True
    standardize: bool = False
    control: GLMControl = field(default_factory=GLMControl)
    offset_id: Union[str,int] = None
    weight_id: Union[str,int] = None
    response_id: Union[str,int] = None
    exclude: list = field(default_factory=list)

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

        _family = family.base
        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = _family.link.inverse(self.link_parameter)

        # shorthand
        self.mu = self.mean_parameter 
        self.eta = self.linear_predictor 

        if family.is_binomial: # isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = _family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):

        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.link_parameter)
        
        # compute working residual
        y = np.asarray(y).reshape(-1)
        r = (y - self.mu) 
        return sample_weight * r * dmu_deta / varmu 

def compute_grad(glm_obj,
                 intercept,
                 coef,
                 design,
                 response,
                 offset=None,
                 scaled_input=False,
                 scaled_output=False,
                 sample_weight=None,
                 norm_weights=False):

    family = glm_obj._family

    if sample_weight is None:
        sample_weight = np.ones(design.shape[0])
        
    if not scaled_input:
        raw_state = GLMState(intercept=intercept,
                             coef=coef)
        scaled_state = design.raw_to_scaled(raw_state)
    else:
        scaled_state = GLMState(intercept=intercept,
                                coef=coef)

    scaled_state.update(design,
                        family,
                        offset=offset)

    saturated_score = scaled_state.logl_score(family,
                                              response, 
                                              sample_weight)
    scaled_score = design.T @ saturated_score
    if not scaled_output:
        score = design.scaler_.T @ scaled_score
    else:
        score = scaled_score
        
    if norm_weights:
        w_sum = sample_weight.sum()
        score /= w_sum
        saturated_score /= w_sum

    return score, saturated_score

@dataclass
class GLMRegularizer(object):

    fit_intercept: bool = False
    standardize: bool = False
    warm_state: dict = field(default_factory=dict)
    ridge_coef: float = 0

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

        D = np.ones(design.X.shape[1] + 1)
        D[0] = 0

        # this will produce coefficients on the raw scale

        if scipy.sparse.issparse(design.X):
            if self.standardize:
                raise ValueError('sklearn Ridge cannot fit without using raw standardized matrix')
            if self.ridge_coef == 0:
                lm = LinearRegression(fit_intercept=self.fit_intercept)
            else:
                lm = Ridge(fit_intercept=self.fit_intercept,
                           alpha=self.ridge_coef)
            lm.fit(design.X, z, sample_weight=w)
            coefnew = lm.coef_
            intnew = lm.intercept_

        else:
            sqrt_w = np.sqrt(w)

            if not self.standardize:
                XW = design.X * sqrt_w[:, None]
            else:
                X = design @ (np.identity(design.shape[1])[:,1:])
                XW = X * sqrt_w[:, None]
                
            if self.fit_intercept:
                D = np.diag(D)
                Wz = sqrt_w * z
                XW = np.concatenate([sqrt_w.reshape((-1,1)), XW], axis=1)
                Q = XW.T @ XW
                if self.ridge_coef != 0:
                    Q += self.ridge_coef * D
                V = XW.T @ Wz
                try:
                    beta = np.linalg.solve(Q, V)
                except LinAlgError as e:
                    if self.control.logging: logging.debug("Error in solve: possible singular matrix, trying pseudo-inverse")
                    if self.ridge_coef != 0:
                        XW = np.vstack([XW, np.sqrt(self.ridge_coef) * D])
                        Wz = np.hstack([Wz, np.zeros(Q.shape[0])])
                    beta = np.linalg.pinv(XW) @ Wz
                coefnew = beta[1:]
                intnew = beta[0]

            else:
                if self.ridge_coef != 0:
                    D = np.diag(D[1:])
                    XW = np.vstack([XW, np.sqrt(self.ridge_coef) * D])
                    z = np.hstack([z, np.zeros(D.shape[0])])
                    sqrt_w = np.hstack([sqrt_w, np.zeros(D.shape[0])])
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
                         nvars=None):
        return GLMRegularizer(fit_intercept=self.fit_intercept)

    def _get_design(self,
                    X,
                    sample_weight):
        return _get_design(X,
                           sample_weight,
                           standardize=self.standardize,
                           intercept=self.fit_intercept)

    def _get_family_spec(self,
                         y):
        return self.family
        # if isinstance(self.family, sm_family.Family):
        #     return GLMFamilySpec(self.family)
        # elif isinstance(self.family, GLMFamilySpec):
        #     return self.family

    def get_data_arrays(self, X, y, check=True):
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check)
    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=None,
            check=True,
            fit_null=True):

        nobs, nvar = X.shape
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)

        X, y, response, offset, weight = self.get_data_arrays(X, y, check=check)

        sample_weight = weight
        self.sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        response = np.asarray(response)

        if self.control is None:
            self.control = GLMControl()
        elif type(self.control) == dict:
            self.control = _parent_dataclass_from_child(GLMControl,
                                                        self.control)
        
        if not hasattr(self, "design_"):
            self.design_ = design = self._get_design(X,
                                                     normed_sample_weight)
        else:
            design = self.design_
            
        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(nvars=design.X.shape[1])
        self.regularizer_ = regularizer

        if warm_state is not None:
            self.regularizer_.warm_state = warm_state
            
        if fit_null or not hasattr(self.regularizer_, 'warm_state'):
            (null_state,
             self.null_deviance_) = self._family.get_null_deviance(
                                        response=response,
                                        sample_weight=sample_weight,
                                        offset=offset,
                                        fit_intercept=self.fit_intercept)


        if (hasattr(self.regularizer_, 'warm_state') and
            self.regularizer_.warm_state):
            state = self.regularizer_.warm_state
        else:
            state = self._family._get_null_state(null_state,
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
         _) = IRLS(regularizer,
                   self._family,
                   design,
                   response,
                   offset,
                   normed_sample_weight,
                   state,
                   obj_function,
                   self.control)

        if self.summarize:
            self._information = self._family.information(state,
                                                         sample_weight)
            
        # checks on convergence and fitted values
        if not converged:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm did not converge")
        if boundary:
            if self.control.logging: logging.debug("Fitting IRLS: algorithm stopped at boundary value")

        self.deviance_ = self._family.deviance(response,
                                               state.mean_parameter,
                                               sample_weight) # not the normalized weights!

        if offset is None:
            offset = np.zeros(y.shape[0])

        self._set_coef_intercept(state)

        if self._family.is_gaussian:
            # usual estimate of sigma^2
            self.dispersion_ = self.deviance_ / (nobs - nvar - self.fit_intercept) 
            self.df_resid_ = nobs - nvar - self.fit_intercept
        else:
            self.dispersion_ = dispersion

        self.state_ = state
        return self
    fit.__doc__ = '''
Fit a GLM.

Parameters
----------

{X}
{y}
{weights}
{summarize}
    
Returns
-------

self: object
        GLM class instance.
        '''.format(**_docstrings)
    
    def predict(self, X, prediction_type='response'):

        linpred = X @ self.coef_ + self.intercept_ # often called eta
        return self._family.predict(linpred, prediction_type=prediction_type)

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
        raw_state = self.design_.scaled_to_raw(state)
        self.coef_ = raw_state.coef
        self.intercept_ = raw_state.intercept

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
    ridge_coef: float = 0

    def _get_regularizer(self,
                         nvars=None):
        return GLMRegularizer(fit_intercept=self.fit_intercept,
                              ridge_coef=self.ridge_coef)

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):

        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    warm_state=warm_state,
                    dispersion=dispersion,
                    check=check)

        weight = self.get_data_arrays(X, y, check=False)[-1]
            
        if self.summarize:
            self.covariance_, self.summary_ = self._summarize(self.exclude,
                                                              self.dispersion_,
                                                              weight,
                                                              X.shape)
        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   X_shape):

        if self.ridge_coef != 0:
            warnings.warn('Detected a non-zero ridge term: variance estimates are taken to be posterior variance estimates')

        # IRLS used normalized weights,
        # this unnormalizes them...

        unscaled_precision_ = self.design_.quadratic_form(self._information,
                                                          transformed=False)
        if self.ridge_coef != 0:
            D = np.ones(unscaled_precision_.shape[0])
            D[0] = 0
            unscaled_precision_ += self.ridge_coef * np.diag(D)
        
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

        if self._family.is_gaussian:
            n, p = X_shape
            self.df_resid_ = n - p - self.fit_intercept
            df_resid = self.df_resid_
        else:
            df_resid = np.inf

        if (df_resid < np.inf):
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     't': T,
                                     'P>|t|': 2 * t_dbn.sf(np.fabs(T),
                                                           df=df_resid)},
                                    index=index)
        else:
            summary_ = pd.DataFrame({'coef':coef,
                                     'std err': SE,
                                     'z': T,
                                     'P>|z|': 2 * normal_dbn.sf(np.fabs(T))},
                                    index=index)
        return covariance_, summary_
        
GLM.__doc__ = '''
Base class to fit a Generalized Linear Model (GLM). Base class for `GLMNet`.

Parameters
----------
{family}
{fit_intercept}
{control_glm}
{response_id}
{weight_id}
{offset_id}
{exclude}

Notes
-----

This is a test

Attributes
----------

{coef_}
{intercept_}
{regularizer_}
{null_deviance_}
{deviance_}
{dispersion_}
{summary_}
{covariance_}
'''.format(**_docstrings)
   
@dataclass
class GaussianGLM(RegressorMixin, GLM):
    pass

@dataclass
class BinomialGLM(ClassifierMixin, GLM):

    family: BinomFamilySpec = field(default_factory=BinomFamilySpec)

    def get_data_arrays(self, X, y, check=True):

        X, y, response, offset, weight = super().get_data_arrays(X, y, check=check)
        encoder = LabelEncoder()
        labels = np.asfortranarray(encoder.fit_transform(response))
        self.classes_ = encoder.classes_
        if len(encoder.classes_) > 2:
            raise ValueError("BinomialGLM expecting a binary classification problem.")
        return X, y, labels, offset, weight

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            warm_state=None,
            dispersion=1,
            check=True):

        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)
            if not self._family.is_binomial:
                msg = f'{self.__class__.__name__} expects a Binomial family.'
                warnings.warn(msg)
                if self.control.logging: logging.warn(msg)

        return super().fit(X,
                           y,
                           sample_weight=sample_weight,
                           regularizer=regularizer,             # last 4 options non sklearn API
                           warm_state=warm_state,
                           dispersion=dispersion,
                           check=check)

    def predict(self, X, prediction_type='class'):

        linpred = X @ self.coef_ + self.intercept_ # often called eta
        pred = self._family.predict(linpred,
                                    prediction_type=prediction_type)
        if prediction_type == 'class':
            pred = self._classes[pred]
        return pred

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

