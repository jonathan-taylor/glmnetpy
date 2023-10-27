import logging
import warnings
from itertools import product

from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.base import (BaseEstimator,
                          clone)
from sklearn.model_selection import (cross_val_predict,
                                     check_cv,
                                     KFold)
from sklearn.model_selection._validation import indexable

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .docstrings import add_dataclass_docstring

from .regularized_glm import (RegGLMControl,
                              RegGLM)
from .glm import (GLM,
                  GLMState,
                  GLMFamilySpec)
from ._utils import _get_data
from .scorer import (PathScorer,
                     plot as plot_cv)

@dataclass
class GLMNetControl(RegGLMControl):

    fdev: float = 1e-5
    logging: bool = False

@dataclass
class GLMNetSpec(object):

    lambda_values: Optional[np.ndarray] = None
    lambda_min_ratio: float = None
    nlambda: int = 100
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None
    fit_intercept: bool = True
    standardize: bool = True
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    control: GLMNetControl = field(default_factory=GLMNetControl)
    regularized_estimator: BaseEstimator = RegGLM
    offset_id: Union[str,int] = None
    weight_id: Union[str,int] = None
    response_id: Union[str,int] = None
    exclude: list = field(default_factory=list)
    
add_dataclass_docstring(GLMNetSpec, subs={'control':'control_glmnet'})

@dataclass
class GLMNet(BaseEstimator,
             GLMNetSpec):

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
                         check=check)

    def _get_family_spec(self, y):
        if isinstance(self.family, sm_family.Family):
            return GLMFamilySpec(self.family)
        elif isinstance(self.family, GLMFamilySpec):
            return self.family

    def fit(self,
            X,
            y,
            sample_weight=None,           # ignored
            regularizer=None,             # last 3 options non sklearn API
            interpolation_grid=None):

        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)

        X, y, response, offset, weight = self._check(X, y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        nobs, nvar = X.shape

        # we use column of y to retrieve optional weight

        sample_weight = weight
        self.normed_sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        
        self.reg_glm_est_ = self.regularized_estimator(
                               lambda_val=self.control.big,
                               family=self.family,
                               alpha=self.alpha,
                               penalty_factor=self.penalty_factor,
                               lower_limits=self.lower_limits,
                               upper_limits=self.upper_limits,
                               fit_intercept=self.fit_intercept,
                               standardize=self.standardize,
                               control=self.control,
                               offset_id=self.offset_id,
                               weight_id=self.weight_id,            
                               response_id=self.response_id,
                               exclude=self.exclude
                               )

        self.reg_glm_est_.fit(X, y, None, fit_null=False) 
        regularizer_ = self.reg_glm_est_.regularizer_

        state, keep_ = self._get_initial_state(X,
                                               y,
                                               self.exclude)

        state.update(self.reg_glm_est_.design_,
                     self._family,
                     offset)

        logl_score = state.logl_score(self._family,
                                      response,
                                      normed_sample_weight)

        score_ = (self.reg_glm_est_.design_.T @ logl_score)[1:]
        pf = regularizer_.penalty_factor
        score_ /= (pf + (pf <= 0))
        score_[self.exclude] = 0
        self.lambda_max_ = np.fabs(score_).max() / max(self.alpha, 1e-3)

        if self.lambda_values is None:
            if self.lambda_min_ratio is None:
                lambda_min_ratio = 1e-2 if nobs < nvar else 1e-4
            else:
                lambda_min_ratio = self.lambda_min_ratio
            self.lambda_values_ = np.exp(np.linspace(
                                          np.log(1),
                                          np.log(lambda_min_ratio),
                                          self.nlambda)
                                         )
            self.lambda_values_ *= self.lambda_max_
        else:
            self.lambda_values_ = np.sort(self.lambda_values)[::-1]
            self.nlambda = self.lambda_values.shape[0]
            self.lambda_min_ratio = (self.lambda_values.min() /
                                     self.lambda_values.max())

        coefs_ = []
        intercepts_ = []
        dev_ratios_ = []
        sample_weight_sum = sample_weight.sum()
        
        (null_fit,
         self.null_deviance_) = self._family.get_null_deviance(response,
                                                               sample_weight,
                                                               offset,
                                                               self.fit_intercept)

        for l in self.lambda_values_:

            if self.control.logging: logging.info(f'Fitting parameter {l}')
            self.reg_glm_est_.lambda_val = regularizer_.lambda_val = l
            self.reg_glm_est_.fit(X,
                                  y,
                                  None, # normed_sample_weight,
                                  regularizer=regularizer_,
                                  check=False,
                                  fit_null=False)

            coefs_.append(self.reg_glm_est_.coef_.copy())
            intercepts_.append(self.reg_glm_est_.intercept_)
            dev_ratios_.append(1 - self.reg_glm_est_.deviance_ / self.null_deviance_)
            if len(dev_ratios_) > 1:
                if isinstance(self.family, sm_family.Gaussian): 
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev * dev_ratios_[-1]:
                        break
                else: # TODO Poisson case
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev:
                        break
            
        self.coefs_ = np.array(coefs_)
        self.intercepts_ = np.array(intercepts_)

        self.summary_ = pd.DataFrame({'Fraction Deviance Explained':dev_ratios_},
                                     index=pd.Series(self.lambda_values_[:len(dev_ratios_)],
                                                     name='lambda'))

        df = (self.coefs_ != 0).sum(1)
        df[0] = 0
        self.summary_.insert(0, 'Degrees of Freedom', df)

        nfit = self.coefs_.shape[0]

        self.lambda_values_ = self.lambda_values_[:nfit]
        
        if interpolation_grid is not None:
            self.coefs_, self.intercepts_ = self.interpolate_coefs(interpolation_grid)
           
        return self
    
    def predict(self,
                X,
                prediction_type='response'):

        if prediction_type not in ['response', 'link']:
            raise ValueError("prediction should be one of 'response' or 'link'")
        
        linear_pred_ = self.coefs_ @ X.T + self.intercepts_[:, None]
        linear_pred_ = linear_pred_.T
        if prediction_type == 'response':
            family = self._family.base
            fits = family.link.inverse(linear_pred_)
        else:
            fits = linear_pred_

        # make return based on original
        # promised number of lambdas
        # pad with last value
        if self.lambda_values is not None:
            nlambda = self.lambda_values.shape[0]
        else:
            nlambda = self.nlambda
        value = np.zeros((fits.shape[0], nlambda), float) * np.nan
        value[:,:fits.shape[1]] = fits
        value[:,fits.shape[1]:] = fits[:,-1][:,None]
        return value
        
    def interpolate_coefs(self,
                          interpolation_grid):

        L = self.lambda_values_
        interpolation_grid = np.clip(interpolation_grid, L.min(), L.max())
        idx_ = interp1d(L, np.arange(L.shape[0]).astype(float))(interpolation_grid)
        coefs_ = []
        intercepts_ = []

        ws = []
        for v_ in idx_:
            v_ceil = int(np.ceil(v_))
            w_ = (v_ceil - v_)
            ws.append(w_)
            if v_ceil > 0:
                coefs_.append(self.coefs_[v_ceil] * (1 - w_) + w_ * self.coefs_[v_ceil-1])
                intercepts_.append(self.intercepts_[v_ceil] * (1 - w_) + w_ * self.intercepts_[v_ceil-1])
            else:
                coefs_.append(self.coefs_[0])
                intercepts_.append(self.intercepts_[0])

        return np.asarray(coefs_), np.asarray(intercepts_)

    def cross_validation_path(self,
                              X,
                              y,
                              cv=10,
                              groups=None,
                              n_jobs=None,
                              verbose=0,
                              fit_params={},
                              pre_dispatch='2*n_jobs',
                              alignment='lambda',
                              scorers=None): # functions of (y, yhat) where yhat is prediction on response scale

        if alignment not in ['lambda', 'fraction']:
            raise ValueError("alignment must be one of 'lambda' or 'fraction'")

        cloned_path = clone(self)
        if alignment == 'lambda':
            fit_params.update(interpolation_grid=self.lambda_values_)
        else:
            if self.lambda_values is not None:
                warnings.warn('Using pre-specified lambda values, not proportional to lambda_max')
            fit_params = None

        X, y, groups = indexable(X, y, groups)
        
        cv = check_cv(cv, y, classifier=False)

        predictions = cross_val_predict(cloned_path,
                                        X,
                                        y,
                                        groups=groups,
                                        cv=cv,
                                        n_jobs=n_jobs,
                                        verbose=verbose,
                                        fit_params=fit_params,
                                        pre_dispatch=pre_dispatch)
        # truncate to the size we got
        predictions = predictions[:,:self.lambda_values_.shape[0]]

        response, offset, weight = self._check(X, y, check=False)[2:]

        # adjust for offset
        # because predictions are just X\beta

        if offset is not None:
            predictions = self._offset_predictions(predictions,
                                                   offset)

        splits = [test for _, test in cv.split(np.arange(X.shape[0]))]

        scorer = PathScorer(predictions=predictions,
                            sample_weight=weight,
                            data=(response, y),
                            splits=splits,
                            family=self._family,
                            index=self.lambda_values_,
                            complexity_order='increasing',
                            compute_std_error=True)

        (self.cv_scores_,
         self.index_best_,
         self.index_1se_) = scorer.compute_scores()

        return predictions, self.cv_scores_

    def plot_cross_validation(self,
                              xvar=None,
                              score=None,
                              ax=None,
                              capsize=3,
                              legend=False,
                              col_min='#909090',
                              ls_min='--',
                              col_1se='#909090',
                              ls_1se='--',
                              c='#c0c0c0',
                              scatter_c='red',
                              scatter_s=None,
                              **plot_args):
        if xvar == 'lambda':
            index = pd.Series(np.log(self.lambda_values_), name=r'$\log(\lambda)$')
        elif xvar == '-lambda':
            index = pd.Series(-np.log(self.lambda_values_), name=r'$-\log(\lambda)$')
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs_).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            index = pd.Index(self.summary_['Fraction Deviance Explained'])
            index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be in ['lambda', '-lambda', 'norm', 'dev']")

        if score is None:
            score = self.family.default_scorers()[0]

        return plot_cv(self.cv_scores_,
                       self.index_best_,
                       self.index_1se_,
                       score=score,
                       index=index,
                       ax=None,
                       capsize=3,
                       legend=False,
                       col_min='#909090',
                       ls_min='--',
                       col_1se='#909090',
                       ls_1se='--',
                       c='#c0c0c0',
                       scatter_c='red',
                       scatter_s=None,
                       **plot_args)

    def plot_coefficients(self,
                          xvar='-lambda',
                          ax=None,
                          legend=False,
                          drop=None,
                          keep=None):

        check_is_fitted(self, ["coefs_", "feature_names_in_"])
        if xvar == '-lambda':
            index = pd.Index(-np.log(self.lambda_values_))
            index.name = r'$-\log(\lambda)$'
        if xvar == 'lambda':
            index = pd.Index(np.log(self.lambda_values_))
            index.name = r'$\log(\lambda)$'
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs_).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            index = pd.Index(self.summary_['Fraction Deviance Explained'])
            index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be one of 'lambda', 'norm', 'dev'")

        soln_path = pd.DataFrame(self.coefs_,
                                 columns=self.feature_names_in_,
                                 index=index)
        if drop is not None:
            soln_path = soln_path.drop(columns=drop)
        if keep is not None:
            soln_path = soln_path.loc[:,keep]
        ax = soln_path.plot(ax=ax, legend=False)
        ax.set_xlabel(index.name)
        ax.set_ylabel(r'Coefficients ($\beta$)')
        ax.axhline(0, c='k', ls='--')

        if legend:
            fig = ax.figure
            if fig.get_layout_engine() is not None:
                warnings.warn('If plotting a legend, layout of figure will be set to "constrained".')
            fig.set_layout_engine('constrained')
            fig.legend(loc='outside right upper')
        return ax

    def _offset_predictions(self,
                            predictions,
                            offset):
        family = self._family.base
        return family.link.inverse(family.link(predictions) +
                                   offset[:,None])
   
    def _get_initial_state(self,
                           X,
                           y,
                           exclude):

        n, p = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor == 0
        keep[exclude] = 0

        coef_ = np.zeros(p)

        if keep.sum() > 0:
            X_keep = X[:,keep]

            glm = GLM(fit_intercept=self.fit_intercept,
                      family=self.family,
                      offset_id=self.offset_id,
                      weight_id=self.weight_id,
                      response_id=self.response_id,
                      control=self.control)
            glm.fit(X_keep, y)
            coef_[keep] = glm.coef_
            intercept_ = glm.intercept_
        else:
            if self.fit_intercept:
                response, offset, weight = self._check(X, y, check=False)[2:]
                state0 = self._family.null_fit(response,
                                               weight,
                                               offset,
                                               self.fit_intercept)
                intercept_ = state0.coef[0] # null state has no intercept
                                            # X a column of 1s
            else:
                intercept_ = 0
        return GLMState(coef_, intercept_), keep.astype(float)

