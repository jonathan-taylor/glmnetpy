import numpy as np

import pytest
rng = np.random.default_rng(0)

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

np_cv_rules = default_converter + numpy2ri.converter

from glmnet import FishNet


def get_glmnet_soln(X,
                    Y,
                    standardize=True,
                    fit_intercept=True):

    with np_cv_rules.context():

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
    G = glmnet(X, Y, family='poisson',
               standardize=standardize, intercept=intercept)
    C = as.matrix(coef(G))
    ''')
        C = rpy.r('C')
    return C.T

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
def test_fishnet(standardize,
                 fit_intercept,
                 n,
                 p):

    X = rng.standard_normal((n, p))
    Y = rng.poisson(20, size=n)
    L = FishNet(standardize=standardize,
                fit_intercept=fit_intercept)

    L.fit(X, Y)
    C = get_glmnet_soln(X,
                        Y,
                        standardize=standardize,
                        fit_intercept=fit_intercept)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10



# import numpy as np
# rng = np.random.default_rng(0)
# n, p = 4, 5
# X = rng.standard_normal((n, p))
# Y = rng.poisson(lam=20, size=n)

# from glmnet import FishNet
# nlambda, lambda_min_ratio = 3, 0.2
# L = FishNet(nlambda=nlambda, lambda_min_ratio=lambda_min_ratio)
# L.fit(X, Y)


# import rpy2
# # %load_ext rpy2.ipython
# # %R -i X,Y,nlambda,lambda_min_ratio

# # + language="R"
# # library(glmnet)
# # G = glmnet(X, Y, family='poisson', nlam=nlambda, lambda.min.ratio=lambda_min_ratio)
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

# L._fit['nin'].max()


