import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y

from .fastnet import MultiFastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._multigaussnet import multigaussnet as _dense
from .._multigaussnet import spmultigaussnet as _sparse

@dataclass
class MultiGaussNet(MultiFastNetMixin):

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

    def _check(self, X, y):
        X, y, response, offset, weight = super()._check(X, y)
        return X, y, np.asfortranarray(response), offset, weight

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):

        if offset is None:
            is_offset = False
        else:
            offset = np.asarray(offset).astype(float)
            response = response - offset # makes a copy, does not modify 
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

