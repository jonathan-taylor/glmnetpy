from dataclasses import dataclass, asdict, field, fields, make_dataclass
   
import numpy as np
import scipy.sparse

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
import statsmodels.api as sm

from ._utils import (_jerr_elnetfit,
                     _obj_function,
                     _dev_function)

from .glmnet_fit import (glmnet_fit,
                         GLMNetResult,
                         DesignSpec,
                         GLMNetSpec,
                         GLMNetControl)

GLMNetPathSpecBase = make_dataclass('GLMNetPathSpec',
                                    [(f.name, f.type, f) for
                                     f in field(GLMNetSpec) if f.name != 'lambda_val'] +
                                    [('lambda_values', np.ndarray)])

@dataclass
class GLMNetPathSpec(GLMNetPathSpecBase):

    pass
