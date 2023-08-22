# IPython log file

import numpy as np
from glmnet.elnet_fit import ElNetSpec
from glmnet.glmnet_path import GLMNetPathSpec, _get_glm_subproblem

n, p = 20, 5
X = np.random.standard_normal((n, p))
y = np.random.standard_normal(n)
lambda_val = 3
elnet = ElNetSpec(X, y, lambda_val)
glmnet_path = GLMNetPathSpec(X, y, np.asarray([lambda_val, 0.5 * lambda_val]))

P = _get_glm_subproblem(glmnet_path, 2.5)
