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

from ..scoring import Scorer

@dataclass
class MultiClassFamily(object):

    def default_scorers(self):
        return [mse_scorer, mae_scorer]

@dataclass
class MultiGaussNet(MultiFastNetMixin):

    response_id: Optional[Union[int,str,list]] = None
    offset_id: Optional[Union[int,str,list]] = None
    standardize_response: bool = False
    _family: MultiClassFamily = field(default_factory=MultiClassFamily)
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

    def _offset_predictions(self,
                            predictions,
                            offset):

        return predictions + offset[:,None,:]

# for CV scores

def _MSE(y, y_hat, sample_weight): 
    return (mean_squared_error(y,
                               y_hat,
                               sample_weight=sample_weight) *
            y_hat.shape[1])

def _MAE(y, y_hat, sample_weight): 
    return (mean_absolute_error(y,
                                y_hat,
                                sample_weight=sample_weight) *
            y_hat.shape[1])


mse_scorer = Scorer(name='Mean Squared Error',
                    score=_MSE,
                    maximize=False)
mae_scorer = Scorer(name='Mean Absolute Error',
                    score=_MAE,
                    maximize=False)

