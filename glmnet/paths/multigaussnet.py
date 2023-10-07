from itertools import product
import logging
import warnings

from typing import Union, Optional
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error)
from .fastnet import MultiFastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._multigaussnet import multigaussnet as _dense
from .._multigaussnet import spmultigaussnet as _sparse

@dataclass
class MultiGaussNet(MultiFastNetMixin):

    response_id: Optional[Union[int,str,list]] = None
    offset_id: Optional[Union[int,str,list]] = None
    standardize_response: bool = False
    _dense = _dense
    _sparse = _sparse

    # private methods

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        # gaussian fit calls it rsq
        # should be `dev` for sumamry

        self._fit['dev'] = self._fit['rsq']
        return super()._extract_fits(X_shape, response_shape)

    def _check(self, X, y, check=True):
        X, y, response, offset, weight = super()._check(X, y, check=check)
        return X, y, np.asfortranarray(response), offset, weight

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):

        if offset is None:
            is_offset = False
            # be sure to copy just in case C++ modifies in place
            # (it does sometimes modify offset)
            response = np.asfortranarray(response.copy())            
        else:
            offset = np.asarray(offset).astype(float)
            response = response - offset # make a copy, do not modify 
            is_offset = True

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        _args['jsd'] = int(self.standardize_response)
        _args['rsq'] = _args['dev']
        del(_args['dev'])
        del(_args['nulldev'])

        return _args

    def _get_scores(self,
                    response,
                    full_y, # ignored by default, used in Cox
                    predictions,
                    sample_weight,
                    test_splits,
                    scorers=[]):

        y = response # shorthand

        scores_ = []

        if scorers is None:
            # create default scorers
            scorers_ = [('Mean Squared Error', mean_squared_error, 'min'),
                        ('Mean Absolute Error', mean_absolute_error, 'min')]

        else:
            scorers_ = scorers
            
        for f, split in enumerate(test_splits):
            preds_ = predictions[split]
            y_ = y[split]
            w_ = sample_weight[split]
            w_ /= w_.mean()
            score_array = np.empty((preds_.shape[1], len(scorers_)), float) * np.nan
            for i, j in product(np.arange(preds_.shape[1]),
                                np.arange(len(scorers_))):
                _, cur_scorer, _ = scorers_[j]
                try:
                    score_array[i, j] = cur_scorer(y_, preds_[:,i], sample_weight=w_)
                except ValueError as e:
                    warnings.warn(f'{cur_scorer} failed on fold {f}, lambda {i}: {e}')                    
                    pass
                    
            scores_.append(score_array)

        return scorers_, np.array(scores_)

