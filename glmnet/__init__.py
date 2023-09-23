from .glm import GLM, GLMControl
from .regularized_glm import RegGLM, RegGLMControl
from .glmnet import GLMNet
from .elnet import ElNet

# fast paths

from .lognet import LogNet
from .gaussnet import GaussNet
from .fishnet import FishNet

# Cox models

from .cox import CoxLM, RegCoxLM, CoxNet

from . import _version
__version__ = _version.get_versions()['version']

