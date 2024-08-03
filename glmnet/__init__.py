from .glm import (GLM,
                  GLMControl,
                  compute_grad,
                  GLMState)
from .regularized_glm import (RegGLM,
                              RegGLMControl)
from .glmnet import GLMNet
from .elnet import ElNet

# fast paths

from .paths import (LogNet,
                    FishNet,
                    GaussNet,
                    MultiGaussNet,
                    MultiClassNet)

# Cox models

from .cox import CoxLM, RegCoxLM, CoxNet

from . import _version
__version__ = _version.get_versions()['version']

