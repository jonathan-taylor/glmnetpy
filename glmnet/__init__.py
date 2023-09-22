from .glm import GLM, GLMControl
from .regularized_glm import RegGLM, RegGLMControl
from .glmnet import GLMNet
from .elnet import ElNet
from .cox import CoxLM, RegCoxLM

from . import _version
__version__ = _version.get_versions()['version']

