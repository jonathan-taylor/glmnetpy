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
from sklearn.linear_model import LinearRegression
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
    control: GLMControl = field(default_factory=GLMControl)
    response_id: Union[str,int] = None
    weight_id: Union[str,int] = None
    offset_id: Union[str,int] = None
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
        return _get_design(X,
                           sample_weight,
                           standardize=False,
                           intercept=self.fit_intercept)

    def _get_family_spec(self,
                         y):
        return self.family
        # if isinstance(self.family, sm_family.Family):
        #     return GLMFamilySpec(self.family)
        # elif isinstance(self.family, GLMFamilySpec):
        #     return self.family

    def _check(self, X, y, check=True):
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
            dispersion=1,
            check=True,
            fit_null=True):

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
            self.design_ = design = self._get_design(X,
                                                     normed_sample_weight)
        else:
            design = self.design_
            
        # for GLM there is no regularization, but this pattern
        # is repeated for GLMNet
        
        # the regularizer stores the warm start

        if regularizer is None:
            regularizer = self._get_regularizer(X)
        self.regularizer_ = regularizer

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

        self.df_resid_ = np.inf
        
        if dispersion is None and self.family.is_gaussian: # Gaussian means identity link, constaint varaiance
            self.dispersion_ = self.deviance_ / (nobs-nvar-self.fit_intercept) 
            n, p = X.shape
            self.df_resid_ = n - p - self.fit_intercept
        else:
            self.dispersion_ = dispersion

        return self
    fit.__doc__ = '''
Fit a GLM.

Parameters
----------

{X}
{y}
{sample_weight}
{regularizer}
{dispersion}
{check}
fit_null: bool
    Fit a null model if no warm state present in the regularizer.

Returns
-------

self: GLMBase
        Fitted GLM.
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
        self.coef_ = state.coef.copy() # this makes a copy -- important to make this copy because `state.coef` is persistent
        if hasattr(self, 'standardize') and self.standardize:
            self.scaling_ = self.design_.scaling_
            self.coef_ /= self.scaling_ 
        self.intercept_ = state.intercept - (self.coef_ * self.design_.centers_).sum()

GLMBase.__doc__ = '''
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

Attributes
----------

{coef_}
{intercept_}
{regularizer_}
{null_deviance_}
{deviance_}
{dispersion_}

'''.format(**_docstrings)



@dataclass
class GLM(GLMBase):

    summarize: bool = False

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            dispersion=1,
            check=True):

        super().fit(X,
                    y,
                    sample_weight=sample_weight,
                    regularizer=regularizer,
                    dispersion=dispersion,
                    check=check)

        weight = self._check(X, y, check=False)[-1]
            
        if self.summarize:
            self.covariance_, self.summary_ = self._summarize(self.exclude,
                                                              self.dispersion_,
                                                              weight,
                                                              self.df_resid_)


        else:
            self.summary_ = self.covariance_ = None
            
        return self

    def _summarize(self,
                   exclude,
                   dispersion,
                   sample_weight,
                   df_resid):

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

    def _check(self, X, y, check=True):

        X, y, response, offset, weight = super()._check(X, y, check=check)
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

