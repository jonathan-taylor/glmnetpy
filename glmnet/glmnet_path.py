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

# setup the GLMNetPathSpec dataclass

_glmnet_path_fields = ([(f.name, f.type, f) for
                       f in fields(GLMNetSpec) if f.name != 'lambda_val'] +
                       [('lambda_values', np.ndarray)])
_glmnet_path_dict = {f[0]:f for f in _glmnet_path_fields}
_final_fields = []
_required = ['X', 'y', 'lambda_values']
for k in _required:
    _final_fields.append(_glmnet_path_dict[k])
_final_fields = _final_fields + [_glmnet_path_dict[k[0]] for k in _glmnet_path_fields if
                                 k[0] not in _required]

GLMNetPathSpecBase = make_dataclass('GLMNetPathSpec',
                                    _final_fields)

@dataclass
class GLMNetPathSpec(GLMNetPathSpecBase):

    pass
