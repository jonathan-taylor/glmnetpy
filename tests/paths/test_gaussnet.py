import numpy as np

import pytest
rng = np.random.default_rng(0)

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

np_cv_rules = default_converter + numpy2ri.converter

from glmnet import GaussNet

def get_glmnet_soln(X,
                    Y,
                    covariance=False,
                    standardize=True,
                    fit_intercept=True):

    with np_cv_rules.context():

        if covariance:
            rpy.r.assign('type.gaussian', "covariance")
        else:
            rpy.r.assign('type.gaussian', "naive")

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
    G = glmnet(X, Y, standardize=standardize, intercept=intercept,
               type.gaussian=type.gaussian)
    C = as.matrix(coef(G))
    ''')
        C = rpy.r('C')
    return C.T

@pytest.mark.parametrize('covariance', [True, False])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
def test_gaussnet(covariance,
                  standardize,
                  fit_intercept,
                  n,
                  p):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept)
    L.fit(X, Y)
    C = get_glmnet_soln(X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10

# # %load_ext rpy2.ipython
# # %R -i X,Y

# X[0]

# # + language="R"
# # library(glmnet)
# # G = glmnet(X, Y)
# # plot(G)
# # G
# # -

# ax = L.plot_coefficients(xvar='norm')

# L.summary_

# # + magic_args="-o C_R,I_R" language="R"
# # C_R = as.matrix(coef(G))[-1,]
# # I_R = as.matrix(coef(G))[1,]
# # -

# np.linalg.norm(C_R.T - L.coefs_) / np.linalg.norm(L.coefs_)

# np.linalg.norm(I_R - L.intercepts_) / np.linalg.norm(L.intercepts_)

# L.coefs_.shape


