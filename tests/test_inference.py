import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.inference import (fixed_lambda_estimator,
                              lasso_inference)
@pytest.mark.parametrize('n', [500])
@pytest.mark.parametrize('p', [103])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_inference(n, p, fit_intercept):
    run_inference(n, p, fit_intercept)

def run_inference(n, p, fit_intercept):

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    Y = np.random.standard_normal(n) * 2
    Df = pd.DataFrame({'response':Y})

    fam = sm.families.Gaussian()
    GN = GLMNet(response_id='response',
                family=fam,
                fit_intercept=fit_intercept)
    GN.fit(X, Df)
    return lasso_inference(GN, 
                           GN.lambda_values_[10],
                           (X[:50], Df.iloc[:50], None),
                           (X, Df, None))

# ncover = 0
# nsel = 0
# niter = 1000
# for i in range(niter):
#     GN = GLMNet(response_id='response',
#                 family=fam,
#                 control=GNcontrol,
#                 fit_intercept=False,
#              )
    
#     X = rng.standard_normal((n, p))
#     Y = rng.standard_normal(n) * 1
#     Df = pd.DataFrame({'response':Y})
#     GN.fit(X, Df)
#     sel_slice = slice(0, 4*n//5)
#     res = lasso_inference(GN, 
#                           GN.lambda_values_[10],
#                           (X[sel_slice], Df.iloc[sel_slice], None),
#                           (X, Df, None),
#                           level=0.8)

#     ncover += ((res['lower'] < 0) & (res['upper'] > 0)).sum()
#     nsel += res.shape[0]

# ncover / nsel

