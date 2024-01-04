from itertools import product
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import warnings

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

        self.scorers = list(set(scorers).union(self.family._default_scorers()))

        score_dict = self._get_scores(self.predictions,
                                      self.sample_weight,
                                      self.splits,
                                      self.scorers)
        df_dict = {}

        for scorer in score_dict.keys():
            val_W = score_dict[scorer]
            val = val_W[:,:,0]
            W = val_W[:,:,1]
            mask = ~np.isnan(val)
            count = mask.sum(1)
            mean_ = np.sum(val * mask * W, 1) / np.sum(mask * W, 1)
            df_dict[scorer.name] = mean_

            if self.compute_std_error:
                resid = val - mean_[:,None]
                std_ = np.sqrt(np.sum(resid**2 * mask * W, 1) / (np.sum(mask * W, 1) * (count-1)))
                df_dict[f'SD({scorer.name})'] = std_
            else:
                std_ = None


        self.cv_scores_ = pd.DataFrame(df_dict,
                                       index=self.index)

        index_best_, index_1se_ = _tune(self.index,
                                        self.scorers,
                                        self.cv_scores_,
                                        complexity_order=self.complexity_order,
                                        compute_std_error=self.compute_std_error)

        return self.cv_scores_, index_best_, index_1se_
    
    def _get_scores(self,
                    predictions,
                    sample_weight,
                    test_splits,
                    scorers):

        response, original_y = self.data
                                      
        scores_ = []
        grouped_weights_ = [sample_weight[split].sum() for split in test_splits]

        final_scores = {}
        for cur_scorer in scorers:
            scores = []
            for i in np.arange(predictions.shape[1]):
                cur_scores = []
                for f, split in enumerate(test_splits):
                    try:
                        if cur_scorer.use_full_data:
                            y = original_y
                        else:
                            y = response
                        val, w = cur_scorer.score_fn(split,
                                                     y,
                                                     predictions[:,i],
                                                     sample_weight=sample_weight)
                    except ValueError as e:
                        warnings.warn(f'Scorer "{cur_scorer.name}" failed on fold {f}, lambda {i}: {e}')                    
                        pass
                    cur_scores.append([val, w])

                if cur_scorer.grouped:
                    cur_scores = np.array(cur_scores)
                else:
                    cur_scores = np.hstack(cur_scores).T
                scores.append(cur_scores)
            scores = np.array(scores) # preds x splits x 2 (the 2 is for value, weight_sum)
            final_scores[cur_scorer] = scores

        return final_scores
    
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
          complexity_order=None,
          compute_std_error=True):

    if (complexity_order is not None
        and complexity_order not in ['increasing', 'decreasing']):
        raise ValueError("complexity order must be in ['increasing', 'decreasing']")

    index_best_ = []
    index_1se_ = []

    npath = cv_scores.shape[0]
    
    for i, scorer in enumerate(scorers):
        picker = {False:np.argmin, True:np.argmax}[scorer.maximize]

        if complexity_order == 'increasing':
            _mean = np.asarray(cv_scores[scorer.name])
        else:
            # in this case indices are taken from the last entry
            # must reshuffle indices below
            _mean = np.asarray(cv_scores[scorer.name].iloc[::-1])

        _best_idx = picker(_mean)
        index_best_.append((scorer.name, index[_best_idx]))

        if compute_std_error:
            _std = np.asarray(cv_scores[f'SD({scorer.name})'])
            if not scorer.maximize:
                _mean_1se = (_mean + _std)[_best_idx]

                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean <=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            else:
                _mean_1se = (_mean - _std)[_best_idx]                    
                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean >=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            if _1se_idx is not None:
                index_1se_.append((scorer.name, index[_1se_idx]))
            else:
                index_1se_.append((scorer.name, np.nan))

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
