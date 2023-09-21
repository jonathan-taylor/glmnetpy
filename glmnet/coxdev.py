from typing import Optional
import numpy as np
import pandas as pd

from coxdev import CoxDeviance

from dataclasses import (dataclass,
                         InitVar)

@dataclass
class CoxDevianceResult(object):

    deviance: float
    grad: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]

