from itertools import product

import numpy as np
from sklearn.model_selection import cross_validate

from glmnet.glmnet import GLMNetEstimator

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

def test_glmnet_R(n=1000, p=50):

    rng = np.random.default_rng(0)

    n, p = 1000, 50
    X = rng.standard_normal((n, p))
    X[:,4] *= 1.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]
    y = rng.standard_normal(n) + X @ beta

    numpy2ri.activate()
    for standardize, intercept, W in product([True, False],
                                             [True, False],
                                             [np.ones(n),
                                              rng.uniform(1, 2, size=(n,))]):

        # with np_cv_rules.context():
        if True:
            G = glmnetR.glmnet(X, y, weights=W, intercept=intercept, standardize=standardize)
            B = glmnetR.predict_glmnet(G, s=2 / np.sqrt(n), type="coef", exact=True, x=X, y=y, weights=W)
            soln_R = baseR.as_numeric(B)
            intercept_R = soln_R[0]
            coef_R = soln_R[1:]

        fac = W.sum() / n
        G = GLMNetEstimator(lambda_val=2 * np.sqrt(n) * fac, 
                            standardize=standardize, 
                            fit_intercept=intercept)
        G.fit(X, y, sample_weight=W)
        D = G._get_design(X, W)
        G.coef_, G.intercept_

        yhat_py = X @ G.coef_ + G.intercept_
        yhat_R = X @ coef_R + intercept_R
        assert np.allclose(yhat_py, yhat_R, atol=1e-5, rtol=1e-5)
        print('match=', np.allclose(yhat_py, yhat_R, atol=1e-5, rtol=1e-5), 'standardize=', standardize, 'intercept=', intercept, W[:2], 'fits close?', np.linalg.norm(yhat_py - yhat_R) / np.linalg.norm(yhat_py))

        assert np.allclose(G.intercept_, intercept_R)
        print('match=', np.allclose(G.intercept_, intercept_R), 'standardize=', standardize, 'intercept=', intercept, W[:2], 'intercept', G.intercept_, intercept_R)
        assert np.allclose(G.coef_, coef_R, atol=1e-5, rtol=1e-5)
        print('match=', np.allclose(G.coef_, coef_R, atol=1e-5, rtol=1e-5), 'standardize=', standardize, 'intercept=', intercept, W[:2], 'coef', np.linalg.norm(G.coef_ - coef_R) / np.linalg.norm(coef_R))

def test_cv():

    rng = np.random.default_rng(0)

    n, p = 1000, 50
    X = rng.standard_normal((n, p))
    X[:,4] *= 1.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]
    y = rng.standard_normal(n) + X @ beta

    G = GLMNetEstimator(lambda_val=2 * np.sqrt(n))
    cross_validate(G, X, y, cv=5)





