import logging

from dataclasses import dataclass, asdict, field, InitVar
from typing import Union, Optional
   
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.base import (BaseEstimator,
                          clone)
from sklearn.model_selection import (cross_val_predict,
                                     check_cv)
from sklearn.model_selection._validation import indexable
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .docstrings import add_dataclass_docstring

from .glmnet import (GLMNetControl,
                     GLMNet)
from .glm import GLM, GLMState


@dataclass
class GLMNetPathControl(GLMNetControl):

    fdev: float = 1e-5
    logging: bool = False

@dataclass
class GLMNetPathSpec(object):

    lambda_values: Optional[np.ndarray] = None
    lambda_fractional: bool = True
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None
    fit_intercept: bool = True
    standardize: bool = True
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMNetPathControl = field(default_factory=GLMNetPathControl)

add_dataclass_docstring(GLMNetPathSpec, subs={'control':'control_glmnet_path'})

@dataclass
class GLMNetPath(BaseEstimator,
                 GLMNetPathSpec):

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            offset=None,
            interpolation_grid=None):

        X, y = check_X_y(X, y,
                         accept_sparse=['csc'],
                         multi_output=False,
                         estimator=self)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = ['X{}'.format(i) for i in range(X.shape[1])]

        nobs, nvar = X.shape

        if self.lambda_values is None:
            self.lambda_fractional = True
            lambda_min_ratio = 1e-2 if nobs < nvar else 1e-4
            self.lambda_values = np.exp(np.linspace(np.log(1),
                                                    np.log(lambda_min_ratio),
                                                    100))

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        self.normed_sample_weight_ = normed_sample_weight = sample_weight / sample_weight.sum()
        
        self.glmnet_est_ = GLMNet(lambda_val=self.control.big,
                                  family=self.family,
                                  alpha=self.alpha,
                                  penalty_factor=self.penalty_factor,
                                  lower_limits=self.lower_limits,
                                  upper_limits=self.upper_limits,
                                  fit_intercept=self.fit_intercept,
                                  standardize=self.standardize,
                                  control=self.control)
        self.glmnet_est_.fit(X, y, normed_sample_weight)
        regularizer_ = self.glmnet_est_.regularizer_

        state, keep_ = self._get_initial_state(X,
                                               y,
                                               normed_sample_weight,
                                               exclude,
                                               offset)
        state.update(self.glmnet_est_.design_,
                     self.family,
                     offset)

        logl_score = state.logl_score(self.family,
                                      y)
        score_ = (self.glmnet_est_.design_.T @ (normed_sample_weight * logl_score))[1:]
        pf = regularizer_.penalty_factor
        score_ /= (pf + (pf <= 0))
        score_[exclude] = 0
        self.lambda_max_ = np.fabs(score_).max() / max(self.alpha, 1e-3)

        if self.lambda_fractional:
            self.lambda_values_ = np.sort(self.lambda_max_ * self.lambda_values)[::-1]
        else:
            self.lambda_values_ = np.sort(self.lambda_values)[::-1]

        coefs_ = []
        intercepts_ = []
        dev_ratios_ = []
        sample_weight_sum = sample_weight.sum()
        
        if self.fit_intercept:
            mu0 = (y * normed_sample_weight).sum() * np.ones_like(y)
        else:
            mu0 = self.family.link.inverse(np.zeros(y.shape, float))
        self.null_deviance_ = self.family.deviance(y,
                                                   mu0,
                                                   freq_weights=sample_weight) # not normed_sample_weight!

        for l in self.lambda_values_:

            if self.control.logging: logging.info(f'Fitting parameter {l}')
            self.glmnet_est_.lambda_val = regularizer_.lambda_val = l
            self.glmnet_est_.fit(X,
                                 y,
                                 normed_sample_weight,
                                 offset=offset,
                                 regularizer=regularizer_,
                                 check=False)

            coefs_.append(self.glmnet_est_.coef_.copy())
            intercepts_.append(self.glmnet_est_.intercept_)
            dev_ratios_.append(1 - self.glmnet_est_.deviance_ * sample_weight_sum / self.null_deviance_)
            if len(dev_ratios_) > 1:
                if isinstance(self.family, sm_family.Gaussian): 
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev * dev_ratios_[-1]:
                        break
                else: # TODO Poisson case
                    if dev_ratios_[-1] - dev_ratios_[-2] < self.control.fdev:
                        break
            
        self.coefs_ = np.array(coefs_)
        self.intercepts_ = np.array(intercepts_)
        self.dev_ratios_ = np.array(dev_ratios_)
        nfit = self.coefs_.shape[0]

        self.lambda_values_ = self.lambda_values_[:nfit]
        
        if interpolation_grid is not None:
            L = self.lambda_values_
            interpolation_grid = np.clip(interpolation_grid, L.min(), L.max())
            idx_ = interp1d(L, np.arange(L.shape[0]))(interpolation_grid)
            coefs_ = []
            intercepts_ = []

            for v_ in idx_:
                v_ceil = int(np.ceil(v_))
                w_ = (v_ceil - v_)
                if v_ceil > 0:
                    coefs_.append(self.coefs_[v_ceil] * w_ + (1 - w_) * self.coefs_[v_ceil-1])
                    intercepts_.append(self.intercepts_[v_ceil] * w_ + (1 - w_) * self.intercepts_[v_ceil-1])
                else:
                    coefs_.append(self.coefs_[0])
                    intercepts_.append(self.intercepts_[0])
            self.coefs_ = np.asarray(coefs_)
            self.intercepts_ = np.asarray(intercepts_)
           
        return self
    
    def predict(self,
                X,
                prediction_type='response',
                lambda_values=None):

        if prediction_type not in ['response', 'link']:
            raise ValueError("prediction should be one of 'response' or 'link'")
        
        linear_pred_ = self.coefs_ @ X.T + self.intercepts_[:, None]
        linear_pred_ = linear_pred_.T
        if prediction_type == 'linear':
            return linear_pred_
        return self.family.link.inverse(linear_pred_)
        
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
            if self.fit_intercept:
                intercept_ = self.family.link(y.mean())
            else:
                intercept_ = 0
        return GLMState(coef_, intercept_), keep.astype(float)

    def cross_validation_path(self,
                              X,
                              y,
                              cv=10,
                              groups=None,
                              n_jobs=None,
                              verbose=0,
                              fit_params={},
                              pre_dispatch='2*n_jobs',
                              alignment='lambda'):

        if alignment not in ['lambda', 'fraction']:
            raise ValueError("alignment must be one of 'lambda' or 'fraction'")

        # within each fold, lambda is fit fractionally

        fractional_path = clone(self)
        fractional_path.lambda_values = self.lambda_values_ / self.lambda_max_
        fractional_path.lambda_fractional = True
        if alignment == 'lambda':
            fit_params.update(interpolation_grid=self.lambda_values_)
        else:
            fit_params = None

        X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=False)
  
        predictions = cross_val_predict(fractional_path, 
                                        X,
                                        y,
                                        groups=groups,
                                        cv=cv,
                                        n_jobs=n_jobs,
                                        verbose=verbose,
                                        fit_params=fit_params,
                                        pre_dispatch=pre_dispatch)

        test_splits = [test for _, test in cv.split(np.arange(X.shape[0]))]

        scores_ = []

        for split in test_splits:
            preds_ = predictions[split]
            y_ = y[split]
            w_ = np.ones_like(y_)
            scores_.append([self.family.deviance(y_,
                                                 preds_[:,i],
                                                 freq_weights=w_) / split.shape[0]
                            for i in range(preds_.shape[1])])
        self.scores_ = np.array(scores_)
        self.dev_cv_mean_ = self.scores_.mean(0)
        self.dev_cv_std_ = self.scores_.std(0, ddof=1) / np.sqrt(self.scores_.shape[0]) # assumes equal size splits!
        self._min_idx = np.argmin(self.dev_cv_mean_)
        self.lambda_min_ = self.lambda_values_[self._min_idx]
        _mean_1se = (self.dev_cv_mean_ + self.dev_cv_std_)[self._min_idx]
        self._1se_idx = max(np.nonzero((self.dev_cv_mean_ <= _mean_1se))[0].min() - 1, 0)
        self.lambda_1se_ = self.lambda_values_[self._1se_idx]

    def plot_coefficients(self,
                          xvar='lambda',
                          ax=None,
                          legend=False,
                          drop=None,
                          keep=None):

        check_is_fitted(self, ["coefs_", "feature_names_in_"])
        if xvar == 'lambda':
            index = pd.Index(-np.log(self.lambda_values_))
            index.name = r'$-\log(\lambda)$'
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs_).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            index = pd.Index(self.dev_ratios_)
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
            fig.legend(loc='outside right upper')
        return ax

    def plot_cross_validation(self,
                              xvar='lambda',
                              ax=None,
                              capsize=3,
                              legend=False,
                              label=None,
                              col_min='k',
                              ls_min='--',
                              col_1se='r',
                              ls_1se='--',
                              **plot_args):

        if label is None:
            label = 'Mean(deviance)'

        check_is_fitted(self, ["dev_cv_mean_", "dev_cv_std_"],
                        msg='This %(name)s is not cross-validated yet. Please run `cross_validation_path` before plotting.')
        if xvar == 'lambda':
            index = pd.Index(-np.log(self.lambda_values_))
            index.name = r'$-\log(\lambda)$'
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs_).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            index = pd.Index(self.dev_ratios_)
            index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be one of 'lambda', 'norm', 'dev'")
        dev_path = pd.DataFrame({label:self.dev_cv_mean_,
                                 'SD':self.dev_cv_std_},
                                index=index)
        ax = dev_path.plot(y=label,
                           kind='line',
                           yerr='SD',
                           legend=legend,
                           **plot_args)
        if xvar == 'lambda':
            l = ax.axvline(-np.log(self.lambda_min_), c=col_min, ls=ls_min, label=r'$\lambda_{\min}$')
            ax.axvline(-np.log(self.lambda_1se_), c=col_1se, ls=ls_1se, label=r'$\lambda_{1SE}$')
        else:
            ax.axvline(np.fabs(self.coefs_[self._min_idx]).sum(), c=col_min, ls=ls_min, label=r'$\lambda_{\min}$')
            ax.axvline(np.fabs(self.coefs_[self._1se_idx]).sum(), c=col_1se, ls=ls_1se, label=r'$\lambda_{1SE}$')
        if legend:
            ax.legend(loc='upper right')
        return ax

