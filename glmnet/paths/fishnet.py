import logging
import warnings

from typing import Literal
from dataclasses import (dataclass,
                         field)
   
import numpy as np

from statsmodels.genmod.families import family as sm_family

from .fastnet import FastNetMixin
from ..glm import GLMFamilySpec
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._fishnet import fishnet as _dense
from .._fishnet import spfishnet as _sparse

@dataclass
class FishNet(FastNetMixin):

    _dense = _dense
    _sparse = _sparse

    # private methods

    def __post_init__(self):
        self._family = GLMFamilySpec(base=sm_family.Poisson())

    def _check(self, X, y, check=True):

        X, y, response, offset, weight = super()._check(X, y, check=check)
        if np.any(response < 0):
            raise ValueError("negative responses encountered;  not permitted for Poisson family")
        response = np.asarray(response, float).copy()
        return X, y, response, offset, weight
    
    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        
        if offset is None:
            offset = 0. * response

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        _args['g'] = np.asfortranarray(offset.reshape((-1,1)).copy())
        return _args

