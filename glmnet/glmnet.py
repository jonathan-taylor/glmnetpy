import logging

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
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             accuracy_score,
                             roc_auc_score)

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

@dataclass
class GLMNetControl(RegGLMControl):

    fdev: float = 1e-5
    logging: bool = False

@dataclass
class GLMNetSpec(object):

    lambda_values: Optional[np.ndarray] = None
    lambda_fractional: bool = True
    alpha: float = 1.0
    lower_limits: float = -np.inf
    upper_limits: float = np.inf
    penalty_factor: Optional[Union[float, np.ndarray]] = None
    fit_intercept: bool = True
    standardize: bool = True
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    control: GLMNetControl = field(default_factory=GLMNetControl)
    regularized_estimator: BaseEstimator = RegGLM
    
add_dataclass_docstring(GLMNetSpec, subs={'control':'control_glmnet'})

@dataclass
class GLMNet(BaseEstimator,
             GLMNetSpec):

    def _check(self,
               X,
               y):

        return check_X_y(X, y,
                         accept_sparse=['csc'],
                         multi_output=False,
                         estimator=self)

    def _get_family_spec(self,
                         y):
        if isinstance(self.family, sm_family.Family):
            return GLMFamilySpec(self.family)
        elif isinstance(self.family, GLMFamilySpec):
            return self.family

    def fit(self,
            X,
            y,
            sample_weight=None,
            regularizer=None,             # last 4 options non sklearn API
            exclude=[],
            offset=None,
            interpolation_grid=None):

        if not hasattr(self, "_family"):
            self._family = self._get_family_spec(y)

        X, y = self._check(X, y)

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
        
        
        self.reg_glm_est_ = self.regularized_estimator(lambda_val=self.control.big,
                                                       family=self.family,
                                                       alpha=self.alpha,
                                                       penalty_factor=self.penalty_factor,
                                                       lower_limits=self.lower_limits,
                                                       upper_limits=self.upper_limits,
                                                       fit_intercept=self.fit_intercept,
                                                       standardize=self.standardize,
                                                       control=self.control)

        self.reg_glm_est_.fit(X, y, normed_sample_weight)
        regularizer_ = self.reg_glm_est_.regularizer_

        state, keep_ = self._get_initial_state(X,
                                               y,
                                               normed_sample_weight,
                                               exclude,
                                               offset)
        state.update(self.reg_glm_est_.design_,
                     self._family,
                     offset)

        logl_score = state.logl_score(self._family,
                                      y,
                                      normed_sample_weight)

        score_ = (self.reg_glm_est_.design_.T @ logl_score)[1:]
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
        
        null_fit, self.null_deviance_ = self._family.get_null_deviance(y,
                                                                       sample_weight,
                                                                       self.fit_intercept)

        for l in self.lambda_values_:

            if self.control.logging: logging.info(f'Fitting parameter {l}')
            self.reg_glm_est_.lambda_val = regularizer_.lambda_val = l
            self.reg_glm_est_.fit(X,
                                  y,
                                  normed_sample_weight,
                                  offset=offset,
                                  regularizer=regularizer_,
                                  check=False)

            coefs_.append(self.reg_glm_est_.coef_.copy())
            intercepts_.append(self.reg_glm_est_.intercept_)
            dev_ratios_.append(1 - self.reg_glm_est_.deviance_ * sample_weight_sum / self.null_deviance_)
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
                prediction_type='response'):

        if prediction_type not in ['response', 'link']:
            raise ValueError("prediction should be one of 'response' or 'link'")
        
        linear_pred_ = self.coefs_ @ X.T + self.intercepts_[:, None]
        linear_pred_ = linear_pred_.T
        if prediction_type == 'linear':
            return linear_pred_
        family = self._family.base
        return family.link.inverse(linear_pred_)
        
    def _get_initial_state(self,
                           X,
                           y,
                           sample_weight,
                           exclude,
                           offset):

        n, p = X.shape
        keep = self.reg_glm_est_.regularizer_.penalty_factor == 0
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
                              alignment='lambda',
                              scorers=None): # functions of (y, yhat) where yhat is prediction on response scale

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

        scorers_, scores_ = self._get_scores(y,
                                             predictions,
                                             test_splits,
                                             scorers=scorers)

        scores_mean_ = scores_.mean(0)
        if isinstance(cv, KFold):
            scores_std_ = scores_.std(0, ddof=1) / np.sqrt(scores_.shape[0]) # assumes equal size splits!

        self.cv_scores_ = pd.DataFrame(scores_mean_,
                                       columns=[name for name, _, _ in scorers_],
                                       index=pd.Series(self.lambda_values_, name='lambda'))

        lambda_best_ = []
        lambda_1se_ = []
        for i, (name, _, pick_best) in enumerate(scorers_):
            picker = {'min':np.argmin, 'max':np.argmax}[pick_best]
            _best_idx = picker(scores_mean_[:,i])
            lambda_best_.append(self.lambda_values_[_best_idx])
            if isinstance(cv, KFold):
                self.cv_scores_[f'SD({name})'] = scores_std_[:,i]
                if pick_best == 'min':
                    _mean_1se = (scores_mean_[:,i] + scores_std_[:,i])[_best_idx]
                    _1se_idx = max(np.nonzero((scores_mean_[:,i] <= _mean_1se))[0].min() - 1, 0)
                elif pick_best == 'max':
                    _mean_1se = (scores_mean_[:,i] - scores_std_[:,i])[_best_idx]                    
                    _1se_idx = max(np.nonzero((scores_mean_[:,i] >= _mean_1se))[0].min() - 1, 0)
                lambda_1se_.append(self.lambda_values_[_1se_idx])

        self.lambda_best_ = pd.Series(lambda_best_, index=[name for name, _, _ in scorers_], name='lambda_best')
        if lambda_1se_:
            self.lambda_1se_ = pd.Series(lambda_1se_, index=[name for name, _, _ in scorers_], name='lambda_1se')
        else:
            self.lambda_1se_ = None

        return self.cv_scores_, self.lambda_best_, self.lambda_1se_
    
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

    def plot_cross_validation(self,
                              xvar='lambda',
                              score=None,
                              ax=None,
                              capsize=3,
                              legend=False,
                              col_min='#c0c0c0',
                              ls_min='--',
                              col_1se='#c0c0c0',
                              ls_1se='--',
                              c='#c0c0c0',
                              scatter_c='red',
                              scatter_s=None,
                              **plot_args):

        if hasattr(self._family, 'base'):
            fam_name = self._family.base.__class__.__name__
        else:
            fam_name = self._family.name
        if score is None:
            score = f'{fam_name} Deviance'

        check_is_fitted(self, ["cv_scores_", "lambda_best_"],
                        msg='This %(name)s is not cross-validated yet. Please run `cross_validation_path` before plotting.')
        if xvar == 'lambda':
            index = pd.Index(-np.log(self.lambda_values_))
            index.name = r'$-\log(\lambda)$'
        elif xvar == 'norm':
            index = pd.Index(np.fabs(self.coefs_).sum(1))
            index.name = r'$\|\beta(\lambda)\|_1$'
        elif xvar == 'dev':
            index = pd.Index(self.summary_['Fraction Deviance Explained'])
            index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be one of 'lambda', 'norm', 'dev'")

        if score not in self.lambda_best_.index:
            raise ValueError(f'Score "{score}" has not been computed in the CV fit.')

        score_path = pd.DataFrame({score: self.cv_scores_[score],
                                  index.name:index})

        ax = score_path.plot.scatter(y=score,
                                     c=scatter_c,
                                     s=scatter_s,
                                     x=index.name,
                                     ax=ax,
                                     zorder=3,
                                     **plot_args)

        if f'SD({score})' in self.cv_scores_.columns:
            have_std = True
            score_path[f'SD({score})'] = self.cv_scores_[f'SD({score})']
            score_path = score_path.set_index(index.name)
            ax = score_path.plot(y=score,
                                 kind='line',
                                 yerr=f'SD({score})',
                                 capsize=capsize,
                                 legend=legend,
                                 c=c,
                                 ax=ax,
                                 **plot_args)
        else:
            score_path = score_path.set_index(index.name)
            ax = score_path.plot(y=score,
                                 kind='line',
                                 legend=legend,
                                 c=c,
                                 ax=ax,
                                 **plot_args)

        ax.set_ylabel(score)
        
        lambda_best = self.lambda_best_[score]
        best_idx = list(self.lambda_values_).index(lambda_best)

        if have_std:
            lambda_1se = self.lambda_1se_[score]
            _1se_idx = list(self.lambda_values_).index(lambda_1se)
        if xvar == 'lambda':
            l = ax.axvline(-np.log(self.lambda_values_[best_idx]), c=col_min, ls=ls_min, label=r'$\lambda_{best}$')
            if have_std:
                ax.axvline(-np.log(self.lambda_values_[_1se_idx]), c=col_1se, ls=ls_1se, label=r'$\lambda_{1SE}$')
        elif xvar == 'norm':
            ax.axvline(np.fabs(self.coefs_[best_idx]).sum(), c=col_min, ls=ls_min, label=r'$\lambda_{best}$')
            if have_std:
                ax.axvline(np.fabs(self.coefs_[_1se_idx]).sum(), c=col_1se, ls=ls_1se, label=r'$\lambda_{1SE}$')
        elif xvar == 'dev':
            dev_ratios = self.summary_['Fraction Deviance Explained']
            ax.axvline(np.fabs(dev_ratios.iloc[best_idx]).sum(), c=col_min, ls=ls_min, label=r'$\lambda_{best}$')
            if have_std:
                ax.axvline(np.fabs(dev_ratios.iloc[_1se_idx]).sum(), c=col_1se, ls=ls_1se, label=r'$\lambda_{1SE}$')
        if legend:
            ax.legend()
        return ax

    def _get_scores(self,
                    y,
                    predictions,
                    test_splits,
                    scorers=[]):

        scores_ = []

        if hasattr(self._family, 'base'):
            fam_name = self._family.base.__class__.__name__
        else:
            fam_name = self._family.__class__.__name__

        if scorers is None:
            # create default scorers
            scorers_ = [(f'{fam_name} Deviance', (lambda y, yhat, sample_weight:
                                                     self._family.deviance(y,
                                                                           yhat,
                                                                           sample_weight) / y.shape[0]),
                         'min'),
                        ('Mean Squared Error', mean_squared_error, 'min'),
                        ('Mean Absolute Error', mean_absolute_error, 'min')]

            if isinstance(self.family, sm_family.Binomial):
                def _accuracy_score(y, yhat, sample_weight): # for binary data classifying at p=0.5, eta=0
                    return accuracy_score(y,
                                          yhat>0.5,
                                          sample_weight=sample_weight,
                                          normalize=True)
                scorers_.extend([('Accuracy', _accuracy_score, 'max'),
                                 ('AUC', roc_auc_score, 'max')])

        else:
            scorers_ = scorers
            
        for split in test_splits:
            preds_ = predictions[split]
            y_ = y[split]
            w_ = np.ones_like(y_)
            scores_.append([[score(y_, preds_[:,i], sample_weight=w_) for _, score, _ in scorers_]
                            for i in range(preds_.shape[1])])
        return scorers_, np.array(scores_)

