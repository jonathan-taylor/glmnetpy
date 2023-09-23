from .glm import GLM, GLMControl
from .regularized_glm import RegGLM, RegGLMControl
from .glmnet import GLMNet
from .elnet import ElNet
from .cox import CoxLM, RegCoxLM

# fast paths

from .lognet import LogNet
from .gaussnet import GaussNet

from . import _version
__version__ = _version.get_versions()['version']

