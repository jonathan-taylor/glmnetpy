import pytest

import numpy as np
from sklearn.model_selection import cross_validate
import statsmodels.api as sm

from glmnet.glmnet import GLMNet

try:
    import rpy2
    has_rpy2 = True

except ImportError:
    has_rpy2 = False

if has_rpy2:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter

    np_cv_rules = default_converter + numpy2ri.converter

    glmnetR = importr('glmnet')
    baseR = importr('base')
    statR = importr('stats')
rng = np.random.default_rng(0)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('alpha', [0, 0.5, 1])
def test_glmnet_R(standardize,
                  intercept,
                  sample_weight,
                  alpha,
                  n=1000,
                  p=50):

    n, p = 1000, 50

    if sample_weight is None:
        sample_weight = np.ones(n)
    else:
        sample_weight = sample_weight(n)
        
    X = rng.standard_normal((n, p))
    X[:,4] *= 4.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]

    with np_cv_rules.context():
        binomial = statR.binomial
        yR = statR.rbinom(n, 1, 0.5)
        sample_weightR = baseR.as_numeric(sample_weight)
        print(yR)
        G = glmnetR.glmnet_path(X,
                                yR,
                                weights=sample_weightR,
                                intercept=intercept,
                                standardize=standardize,
                                family=binomial,
                                alpha=alpha)
        B = glmnetR.predict_glmnet(G,
                                   s=0.5 / np.sqrt(n),
                                   type="coef",
                                   exact=True,
                                   x=X,
                                   y=yR,
                                   weights=sample_weightR)
        soln_R = baseR.as_numeric(B)
        intercept_R = soln_R[0]
        coef_R = soln_R[1:]

    G = GLMNet(lambda_val=0.5 / np.sqrt(n),
               family=sm.families.Binomial(),
               alpha=alpha,
               standardize=standardize, 
               fit_intercept=intercept)
    y = np.asarray(yR)
    G.fit(X, y, sample_weight=sample_weight)

    soln_py = np.hstack([G.intercept_, G.coef_])
    soln_R = np.hstack([intercept_R, coef_R])    
    yhat_py = G.design_ @ soln_py
    yhat_R = G.design_ @ soln_R 

    fit_match = np.allclose(yhat_py, yhat_R, atol=1e-5, rtol=1e-5)
    intercept_match = np.allclose(G.intercept_, intercept_R)
    coef_match = np.allclose(G.coef_, coef_R, atol=1e-5, rtol=1e-5)

    print(f'fit: {fit_match}, intercept: {intercept_match}, coef:{coef_match}')

    assert fit_match and intercept_match and coef_match

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_cv(standardize,
            fit_intercept,
            n=1000,
            p=50):

    X = rng.standard_normal((n, p))
    X[:,4] *= 1.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]
    y = (rng.standard_normal(n) + X @ beta > 0)

    G = GLMNet(lambda_val=0.5 / np.sqrt(n),
               family=sm.families.Binomial(),
               fit_intercept=fit_intercept,
               standardize=standardize)
    cross_validate(G, X, y, cv=5)





