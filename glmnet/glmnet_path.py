from typing import Union, Optional
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
                         GLMNetSpec,
                         GLMNetControl)

from .elnet_fit import (BaseSpec,
                        DesignSpec,
                        _set_limits,
                        _set_vp,
                        _set_design)

@dataclass
class GLMNetPathSpec(BaseSpec):

    lambda_fracs: np.ndarray
    alpha: float = 1.0
    lower_limits: Union[float, np.ndarray] = -np.inf
    upper_limits: Union[float, np.ndarray] = np.inf
    exclude: list = field(default_factory=list)
    penalty_factor: Optional[Union[np.ndarray, float]] = None
    weights: Optional[np.ndarray] = None
    intercept: bool = True
    offset: np.ndarray = None
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMNetControl = field(default_factory=GLMNetControl)

    def __post_init__(self):

        if self.control is None:
            self.control = GLMNetControl()
        elif type(self.control) == dict:
            self.control = GLMNetControl(**self.control)

        if self.exclude is None:
            self.exclude = []
            
        if self.offset is None:
            self.is_offset = False
            self.offset = np.zeros(self.y.shape)
        else:
            self.is_offset = True

        _set_limits(self)
        _set_vp(self)
        _set_design(self)

def _get_glm_subproblem(glm_path,
                        lambda_val):

    glm_dict = asdict(glm_path)
    glm_dict['lambda_val'] = lambda_val
    del(glm_dict['lambda_fracs'])
    glm_dict['X'] = glm_path.design
    return GLMNetSpec(**glm_dict)

# # setup the GLMNetPathSpec dataclass

# _glmnet_path_fields = ([(f.name, f.type, f) for
#                        f in fields(GLMNetSpec) if f.name != 'lambda_val'] +
#                        [('lambda_values', np.ndarray)])
# _glmnet_path_dict = {f[0]:f for f in _glmnet_path_fields}
# _final_fields = []
# _required = ['X', 'y', 'lambda_values']
# for k in _required:
#     _final_fields.append(_glmnet_path_dict[k])
# _final_fields = _final_fields + [_glmnet_path_dict[k[0]] for k in _glmnet_path_fields if
#                                  k[0] not in _required]

# GLMNetPathSpecBase = make_dataclass('GLMNetPathSpec',
#                                     _final_fields)

# @dataclass
# class GLMNetPathSpec(GLMNetPathPreSpec,PathSpec):

#     pass
