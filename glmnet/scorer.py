from itertools import product
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.model_selection import (cross_val_predict,
                                     check_cv,
                                     KFold)

from .glm import GLMFamilySpec

@dataclass
class PathScorer(object):

    data: tuple
    predictions: np.ndarray
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    sample_weight: np.ndarray = None
    splits: list = field(default_factory=list)
    compute_std_error: bool = True
    index: pd.Series = None
    complexity_order: str = 'increasing'
    
    def __post_init__(self):

        if type(self.index) == np.ndarray:
            self.index = pd.Series(self.index,
                                   index=np.arange(self.index.shape[0]))

    def compute_scores(self,
                       scorers=[]):

        self.scorers = list(set(scorers).union(self.family.default_scorers()))

        response, y_arg = self.data

        scores_ = self._get_scores(response,
                                    y_arg,
                                    self.predictions,
                                    self.sample_weight,
                                    self.splits,
                                    self.scorers)

        weight = self.sample_weight
        wsum = np.array([weight[split].sum() for split in self.splits])
        wsum = wsum[:, None, None]
        mask = ~np.isnan(scores_)
        scores_mean_ = (np.sum(scores_ * mask * wsum, 0) /
                        np.sum(mask * wsum, 0))

        if self.compute_std_error:
            count = mask.sum(0)
            resid_ = scores_ - scores_mean_[None,:]
            scores_std_ = np.sqrt(np.sum(resid_**2 *
                                         mask * wsum, 0) /
                                  (np.sum(mask * wsum, 0) * (count - 1)))
        else:
            scores_std_ = None

        self.cv_scores_ = pd.DataFrame(scores_mean_,
                                       columns=[name for
                                                name, _, _, _ in self.scorers],
                                       index=self.index)

        for i, (name, _, _, _) in enumerate(self.scorers):
            self.cv_scores_[f'SD({name})'] = scores_std_[:,i]
        
        index_best_, index_1se_ = _tune(self.index,
                                        self.scorers,
                                        self.cv_scores_,
                                        scores_mean_,
                                        scores_std_,
                                        complexity_order=self.complexity_order,
                                        compute_std_error=self.compute_std_error)
        return self.cv_scores_, index_best_, index_1se_
    
    def _get_scores(self,
                    response,
                    full_y, # ignored by default, used in Cox
                    predictions,
                    sample_weight,
                    test_splits,
                    scorers):

        y = response # shorthand

        scores_ = []
           
        for f, split in enumerate(test_splits):
            preds_ = predictions[split]
            y_ = y[split]
            w_ = sample_weight[split]
            w_ = w_ / w_.mean()
            score_array = np.empty((preds_.shape[1], len(scorers)), float) * np.nan
            for i, j in product(np.arange(preds_.shape[1]),
                                np.arange(len(scorers))):
                _, cur_scorer, _, use_full = scorers[j]
                try:
                    if not use_full:
                        score_array[i, j] = cur_scorer(y_,
                                                       preds_[:,i],
                                                       sample_weight=w_)
                    else:
                        score_array[i, j] = cur_scorer(full_y,
                                                       split,
                                                       preds_[:,i],
                                                       sample_weight=w_)
                except ValueError as e:
                    warnings.warn(f'{cur_scorer} failed on fold {f}, lambda {i}: {e}')                    
                    pass
                    
            scores_.append(score_array)

        return np.array(scores_)
    
def plot(cv_scores,
         index_best,
         index_1se,
         index=None,
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

    if score not in index_best:
        raise ValueError(f'Score "{score}" was not computed in the CV fit.')

    score_path = pd.DataFrame({score: cv_scores[score]}).reset_index()
    score_path[index.name] = index

    ax = score_path.plot.scatter(y=score,
                                 c=scatter_c,
                                 s=scatter_s,
                                 x=index.name,
                                 ax=ax,
                                 zorder=3,
                                 **plot_args)

    if f'SD({score})' in cv_scores.columns:
        have_std = True
        # use values as the index of cv_scores might be different
        score_path[f'SD({score})'] = cv_scores[f'SD({score})'].values
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
    
    # add the lines indicating best and 1se choice

    if score in index_best.index:
        _best_idx = list(cv_scores.index).index(index_best[score])
        ax.axvline(index[_best_idx],
                   c=col_min,
                   ls=ls_min,
                   label=r'Best')
        
    if score in index_1se.index:
        _1se_idx = list(cv_scores.index).index(index_1se[score])
        ax.axvline(index[_1se_idx],
                   c=col_min,
                   ls=ls_min,
                   label=r'1SE')
        
    if legend:
        ax.legend()
    return ax


def _tune(index,
          scorers,
          cv_scores,
          scores_mean,
          scores_std,
          complexity_order=None,
          compute_std_error=True):

    if (complexity_order is not None
        and complexity_order not in ['increasing', 'decreasing']):
        raise ValueError("complexity order must be in ['increasing', 'decreasing']")

    index_best_ = []
    index_1se_ = []

    npath = cv_scores.shape[0]
    
    for i, (name, _, pick_best, _) in enumerate(scorers):
        picker = {'min':np.argmin, 'max':np.argmax}[pick_best]

        if complexity_order == 'increasing':
            _mean = scores_mean[:,i]
        else:
            # in this case indices are taken from the last entry
            # must reshuffle indices below
            _mean = scores_mean[:,i][::-1]

        _best_idx = picker(_mean)
        index_best_.append((name, index[_best_idx]))

        if compute_std_error:
            _std = scores_std[:,i]
            if pick_best == 'min':
                _mean_1se = (_mean + _std)[_best_idx]

                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean <=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            elif pick_best == 'max':
                _mean_1se = (_mean - _std)[_best_idx]                    
                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean >=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            if _1se_idx is not None:
                index_1se_.append((name, index[_1se_idx]))
            else:
                index_1se_.append((name, np.nan))

    index_best_ = pd.Series([v for _, v in index_best_],
                            index=[n for n, _ in index_best_],
                            name='index_best')

    if complexity_order == 'decreasing':
        index_best_ = npath - 1 - index_best_

    if len(index_1se_) > 0:
        index_1se_ = pd.Series([v for _, v in index_1se_],
                                index=[n for n, _ in index_1se_],
                               name='index_1se')
        if complexity_order == 'decreasing':
            index_1se_ = npath - 1 - index_1se_

    return index_best_, index_1se_
