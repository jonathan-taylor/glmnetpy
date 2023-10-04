import numpy as np

import pytest
rng = np.random.default_rng(0)

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

np_cv_rules = default_converter + numpy2ri.converter

from glmnet import MultiGaussNet

def get_glmnet_soln(X,
                    Y,
                    standardize_response=True,
                    standardize=True,
                    fit_intercept=True):

    with np_cv_rules.context():

        if standardize_response:
            rpy.r.assign('standardize.response', True)
        else:
            rpy.r.assign('standardize.response', False)

        if standardize:
            rpy.r.assign('standardize', True)
        else:
            rpy.r.assign('standardize', False)

        if fit_intercept:
            rpy.r.assign('intercept', True)
        else:
            rpy.r.assign('intercept', False)

        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        rpy.r('''
    library(glmnet)
    X = as.matrix(X)
    Y = as.matrix(Y)    
    print(dim(X))
    print(dim(Y))
    G = glmnet(X, Y, standardize=standardize, intercept=intercept,
               family='mgaussian',
               standardize.response=standardize.response)
    ''')
        rpy.r('C = coef(G)')

        C = []
        for v in range(Y.shape[1]):
            rpy.r(f'C{v} = as.matrix(C$y{v+1})')
            C.append(rpy.r(f'C{v}'))
    C = np.asarray(C)
    return np.transpose(C, [2, 1, 0])

@pytest.mark.parametrize('standardize_response', [True, False])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
# R fails for q=1
@pytest.mark.parametrize('q', [3,1][:1])
def test_multigaussnet(standardize_response,
                       standardize,
                       fit_intercept,
                       n,
                       p,
                       q):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    L = MultiGaussNet(standardize_response=standardize_response,
                      standardize=standardize,
                      fit_intercept=fit_intercept)
    L.fit(X, Y)
    C = get_glmnet_soln(X,
                        Y,
                        standardize_response=standardize_response,
                        standardize=standardize,
                        fit_intercept=fit_intercept)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10

