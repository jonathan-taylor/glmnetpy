import logging
import warnings

from typing import Literal
from dataclasses import dataclass, field
   
import numpy as np
import pandas as pd
import scipy.sparse

from .fastnet import FastNetMixin
from ..docstrings import (make_docstring,
                          add_dataclass_docstring)

from .._gaussnet import gaussnet as _dense
from .._gaussnet import spgaussnet as _sparse

@dataclass
class GaussNet(FastNetMixin):

    covariance : bool = None

    _dense = _dense
    _sparse = _sparse

    # private methods

    def _extract_fits(self,
                      X_shape,
                      y_shape):
        self._fit['dev'] = self._fit['rsq'] # gaussian fit calls it rsq
        return super()._extract_fits(X_shape,
                                     y_shape)
        
    def _wrapper_args(self,
                      design,
                      y,
                      sample_weight,
                      offset,
                      exclude=[]):

        if offset is None:
            is_offset = False
        else:
            offset = np.asarray(offset).astype(float)
            y = y - offset # makes a copy, does not modify y
            is_offset = True

        # compute nulldeviance

        ybar = (y * sample_weight).sum() / sample_weight.sum()
        nulldev = ((y - ybar)**2 * sample_weight).sum() / sample_weight.sum()

        if nulldev == 0:
            raise ValueError("y is constant; GaussNet fails at standardization step")

        _args = super()._wrapper_args(design,
                                      y.copy(), # it will otherwise be scaled
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # add 'ka' 
        if self.covariance is None:
            nvars = design.X.shape[1]
            if nvars < 500:
                self.covariance = True
            else:
                self.covariance = False

        _args['ka'] = {True:1,
                       False:2}[self.covariance]

        # Gaussian calls it rsq
        _args['rsq'] = _args['dev']
        del(_args['dev'])

        # doesn't use nulldev
        del(_args['nulldev'])
        return _args
