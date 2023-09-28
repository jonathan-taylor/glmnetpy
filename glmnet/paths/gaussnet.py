import logging
import warnings

from typing import Literal
from dataclasses import dataclass, field
   
import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.utils import check_X_y

from ._gaussnet import gaussnet as _dense
from ._gaussnet import spgaussnet as _sparse

from .base import _get_design
from .glmnet import GLMNet
from .elnet import (_check_and_set_limits,
                    _check_and_set_vp,
                    _design_wrapper_args)

from ._utils import _jerr_elnetfit
from .docstrings import (make_docstring,
                         add_dataclass_docstring)
from .fastnet import FastNetMixin

@dataclass
class GaussNet(FastNetMixin):

    type_gaussian: Literal['covariance', 'naive'] = None

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
                                      y,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # add 'ka' 
        if self.type_gaussian is None:
            nvars = design.X.shape[1]
            if nvars < 500:
                self.type_gaussian = 'covariance'
            else:
                self.type_gaussian = 'naive'

        _args['ka'] = {'covariance':1,
                       'naive':2}[self.type_gaussian]

        # Gaussian calls it rsq
        _args['rsq'] = _args['dev']
        del(_args['dev'])

        # doesn't use nulldev
        del(_args['nulldev'])
        return _args
