from itertools import product
import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y
from sklearn.metrics import (accuracy_score,
                             zero_one_loss,
                             log_loss)

from .fastnet import MultiFastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._lognet import lognet as _dense
from .._lognet import splognet as _sparse

from ..scoring import Scorer

@dataclass
class MultiClassFamily(object):

    def default_scorers(self):

        return [accuracy_scorer,
                misclass_scorer,
                deviance_scorer]

@dataclass
class MultiClassNet(MultiFastNetMixin):

    standardize_response: bool = False
    grouped: bool = False
    univariate_beta: bool = True
    type_logistic: Literal['Newton', 'modified_Newton'] = 'Newton'
    _family: MultiClassFamily = field(default_factory=MultiClassFamily)
    _dense = _dense
    _sparse = _sparse

    def predict(self,
                X,
                prediction_type='response' # ignored except checking valid
                ):

        value = super().predict(X, prediction_type='link')
        if prediction_type == 'response':
            _max = value.max(-1)
            value = value - _max[:,:,None]
            exp_value = np.exp(value)
            value = exp_value / exp_value.sum(-1)[:,:,None]
        return value
        
    # private methods

    def _offset_predictions(self,
                            predictions,
                            offset):
        value = np.log(predictions) + offset[:,None,:]
        _max = value.max(-1)
        value = value - _max[:,:,None]
        exp_value = np.exp(value)
        value = exp_value / exp_value.sum(-1)[:,:,None]
        return value

    def _check(self, X, y, check=True):

        X, y, response, offset, weight = super()._check(X, y, check=check)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = np.asfortranarray(encoder.fit_transform(response.reshape((-1,1))))
        self.categories_ = encoder.categories_[0]
        return X, y, y_onehot, offset, weight

    def _extract_fits(self,
                      X_shape,
                      response_shape):
        # center the intercepts -- any constant
        # added does not affect class probabilities
        
        self._fit['a0'] = self._fit['a0'] - self._fit['a0'].mean(0)[None,:]
        return super()._extract_fits(X_shape, response_shape)

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        if offset is None:
            offset = response * 0.
        if offset.shape != response.shape:
            raise ValueError('offset shape should match one-hot response shape')
        offset = np.asfortranarray(offset.copy())

        # add 'kopt' 
        _args['kopt'] = {'Newton':0,
                         'modified_Newton':1}[self.type_logistic]
        # if grouped, we set kopt to 2
        if self.grouped:
            _args['kopt'] = 2

        # add 'g'
        _args['g'] = offset

        # take care of weights
        _args['y'] = np.asfortranarray(_args['y'] * sample_weight[:,None])

        # remove w
        del(_args['w'])

        return _args

# for CV scores

def _misclass(y, p_hat, sample_weight): 
    return zero_one_loss(np.argmax(y, -1),
                         np.argmax(p_hat, -1),
                         sample_weight=sample_weight,
                         normalize=True)

def _accuracy_score(y, p_hat, sample_weight): 
    return accuracy_score(np.argmax(y, -1),
                          np.argmax(p_hat, -1),
                          sample_weight=sample_weight,
                          normalize=True)

def _deviance(y, p_hat, sample_weight): 
    return 2 * log_loss(y, p_hat, sample_weight=sample_weight)

misclass_scorer = Scorer(name='Misclassification Error',
                         score=_misclass,
                         maximize=False)

accuracy_scorer = Scorer(name='Accuracy',
                         score=_accuracy_score,
                         maximize=True)

deviance_scorer = Scorer(name="Multinomial Deviance",
                         score=_deviance,
                         maximize=False)
