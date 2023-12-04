import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from glmnet.regularized_glm import RegGLM

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

def nonuniform_(n):
    W = rng.uniform(0, 1, size=(n,))
    W[:n//2] *= 2
    return W

rng = np.random.default_rng(0)

ifrpy = pytest.mark.skipif(not has_rpy2, reason='requires rpy2')
standardize = pytest.mark.parametrize('standardize', [True, False])
fit_intercept = pytest.mark.parametrize('fit_intercept', [True, False])
sample_weight = pytest.mark.parametrize('sample_weight', [np.ones, nonuniform_])
alpha = pytest.mark.parametrize('alpha', [0, 0.4, 1])
path = pytest.mark.parametrize('path', [True, False])

@ifrpy
@standardize
@fit_intercept
@sample_weight
@alpha
@path
def test_glmnet(standardize,
                fit_intercept,
                sample_weight,
                alpha,
                path,
                n=1000,
                p=50):
    '''
    compare to glmnet:::glmnet
    '''
    n, p = 1000, 50

    if sample_weight is None:
        sample_weight = np.ones(n)
    else:
        sample_weight = sample_weight(n)
    sample_weight /= sample_weight.sum()
    
    X = rng.standard_normal((n, p))
    X[:,4] *= 4.5
    X[:,2] *= 1.3
    X[:,0] *= 0.6
    X[:,1] *= 0.8
    beta = np.zeros(p)
    beta[:2] = [1,2]
    y = 2 * rng.standard_normal(n) + X @ beta

    if alpha < 1:
        # standardize y like glmnet
        w_mean = (y * sample_weight).sum() / sample_weight.sum()
        w_std = np.sqrt(((y - w_mean)**2 * sample_weight).sum() / sample_weight.sum())
        y /= w_std
        
    with np_cv_rules.context():
        if path:
            Gfit = glmnetR.glmnet_path
        else:
            Gfit = glmnetR.glmnet
        G = Gfit(X,
                 y,
                 weights=sample_weight,
                 intercept=fit_intercept,
                 standardize=standardize,
                 alpha=alpha)
        B = glmnetR.predict_glmnet(G, s=2/np.sqrt(n), type="coef", exact=True, x=X, y=y, weights=sample_weight)
        soln_R = baseR.as_numeric(B)
        intercept_R = soln_R[0]
        coef_R = soln_R[1:]

    G = RegGLM(lambda_val=2/np.sqrt(n),
               alpha=alpha,
               standardize=standardize, 
               fit_intercept=fit_intercept,
               response_id='response',
               weight_id='weight')

    df = pd.DataFrame({'response':y, 'weight':sample_weight})
    G.fit(X, df)

    soln_py = np.hstack([G.intercept_, G.coef_])
    soln_R = np.hstack([intercept_R, coef_R])    
    yhat_py = G.design_ @ soln_py
    yhat_R = G.design_ @ soln_R 

    fit_match = np.allclose(yhat_py, yhat_R, atol=1e-3, rtol=1e-3)
    intercept_match = np.fabs(G.intercept_ - intercept_R) < 1e-3
    coef_match = np.fabs(coef_R - G.coef_).max() / np.linalg.norm(G.coef_) < 1e-3

    print(f'fit: {fit_match}, intercept: {intercept_match}, coef:{coef_match}')
    print('intercepts:', intercept_R, G.intercept_)
    assert fit_match and intercept_match and coef_match

@standardize
@fit_intercept
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
    y = rng.standard_normal(n) + X @ beta

    G = RegGLM(lambda_val=2 * np.sqrt(n),
               fit_intercept=fit_intercept,
               standardize=standardize)
    cross_validate(G, X, y, cv=5)





