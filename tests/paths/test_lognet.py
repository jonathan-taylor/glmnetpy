import numpy as np

import pytest
rng = np.random.default_rng(0)

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

np_cv_rules = default_converter + numpy2ri.converter

from glmnet import LogNet

def get_glmnet_soln(X,
                    Y,
                    modified_newton,
                    standardize=True,
                    fit_intercept=True):

    with np_cv_rules.context():

        if modified_newton:
            rpy.r.assign('type.logistic', "modified.Newton")
        else:
            rpy.r.assign('type.logistic', "Newton")

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
    y = as.integer(Y=='A')
    G = glmnet(X, y, family='binomial',
               standardize=standardize, intercept=intercept,
               type.logistic=type.logistic)
    C = as.matrix(coef(G))
    ''')
        C = rpy.r('C')
    return C.T

@pytest.mark.parametrize('modified_newton', [True, False])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
def test_lognet(modified_newton,
                standardize,
                fit_intercept,
                n,
                p):

    X = rng.standard_normal((n, p))
    Y = rng.choice(['A', 'B'], size=n)
    L = LogNet(modified_newton=modified_newton,
               standardize=standardize,
               fit_intercept=fit_intercept)

    L.fit(X, Y)
    C = get_glmnet_soln(X,
                        Y,
                        modified_newton=modified_newton,
                        standardize=standardize,
                        fit_intercept=fit_intercept)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10

