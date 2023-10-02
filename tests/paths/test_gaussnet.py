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
                    fit_intercept=True,
                    exclude=[],
                    df_max=None,
                    nlambda=None,
                    lambda_min_ratio=None,
                    lower_limits=None,
                    upper_limits=None,
                    penalty_factor=None,
                    offset=None,
                    weights=None):

    with np_cv_rules.context():

        args = []

        if df_max is not None:
            rpy.r.assign('dfmax', df_max)
            args.append('dfmax=dfmax')
        if weights is not None:
            rpy.r.assign('weights', weights)
            args.append('weights=weights')

        if offset is not None:
            rpy.r.assign('offset', offset)
            args.append('offset=offset')

        if lambda_min_ratio is not None:
            rpy.r.assign('lambda.min.ratio', lambda_min_ratio)
            args.append('lambda.min.ratio=lambda.min.ratio')

        if lower_limits is not None:
            rpy.r.assign('lower.limits', lower_limits)
            args.append('lower.limits=lower.limits')
        if upper_limits is not None:
            rpy.r.assign('upper.limits', upper_limits)
            args.append('upper.limits=upper.limits')

        if penalty_factor is not None:
            rpy.r.assign('penalty.factor', penalty_factor)
            args.append('penalty.factor=penalty.factor')

        if nlambda is not None:
            rpy.r.assign('nlambda', nlambda)
            args.append('nlambda=nlambda')

        if covariance:
            rpy.r.assign('type.gaussian', "covariance")
            args.append('type.gaussian="covariance"')
        else:
            rpy.r.assign('type.gaussian', "naive")
            args.append('type.gaussian="naive"')

        if standardize:
            rpy.r.assign('standardize', True)
        else:
            rpy.r.assign('standardize', False)
        args.append('standardize=standardize')

        if fit_intercept:
            rpy.r.assign('intercept', True)
        else:
            rpy.r.assign('intercept', False)
        args.append('intercept=intercept')

        rpy.r.assign('exclude', np.array(exclude))
        args.append('exclude=exclude')

        args = ','.join(args)

        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
        library(glmnet)
        G = glmnet(X, Y, {args})
        C = as.matrix(coef(G))
            '''
        print(cmd)

        rpy.r(cmd)
        C = rpy.r('C')
    return C.T

# weight functions

def sample1(n):
    return rng.uniform(0, 1, size=n)

def sample2(n):
    V = sample1(n)
    V[:n//2] = 0


# @pytest.mark.parametrize('sample_weight', [None, np.ones, sample1, sample2])
# @pytest.mark.parametrize('df_max', [None, 5])
# @pytest.mark.parametrize('exclude', [[], [1,2,3]])
# @pytest.mark.parametrize('lower_limits', [-1e-3,-1, None][1:])
# @pytest.mark.parametrize('covariance', [True, False])
# @pytest.mark.parametrize('standardize', [True, False])
# @pytest.mark.parametrize('fit_intercept', [True, False])
# @pytest.mark.parametrize('nlambda', [None, 20])
# @pytest.mark.parametrize('lambda_min_ratio', [None,0.02])
# @pytest.mark.parametrize('n', [1000,50])
# @pytest.mark.parametrize('p', [10,100])
# def test_gaussnet(covariance,
#                   standardize,
#                   fit_intercept,
#                   exclude,
#                   lower_limits,
#                   nlambda,
#                   lambda_min_ratio,
#                   sample_weight,
#                   df_max,
#                   n,
#                   p):

#     if lower_limits is not None:
#         lower_limits = np.ones(p) * lower_limits
#     X = rng.standard_normal((n, p))
#     Y = rng.standard_normal(n)
#     L = GaussNet(covariance=covariance,
#                  standardize=standardize,
#                  fit_intercept=fit_intercept,
#                  lambda_min_ratio=lambda_min_ratio,
#                  df_max=df_max)
#     if nlambda is not None:
#         L.nlambda = nlambda

#     if sample_weight is not None:
#         weights = sample_weight(n)
#     else:
#         weights = None

#     L.fit(X,
#           Y,
#           exclude=exclude,
#           sample_weight=weights)

#     C = get_glmnet_soln(X,
#                         Y,
#                         covariance=covariance,
#                         standardize=standardize,
#                         fit_intercept=fit_intercept,
#                         lower_limits=lower_limits,
#                         exclude=exclude,
#                         weights=weights,
#                         nlambda=nlambda,
#                         lambda_min_ratio=lambda_min_ratio,
#                         df_max=df_max)


#     assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
#     if fit_intercept:
#         assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10


# @pytest.mark.parametrize('limits', [(-1, np.inf), (-np.inf, 1),
#                                     (-np.inf, np.inf),
#                                     (-1, 1),
#                                     (0, 1)])
# @pytest.mark.parametrize('penalty_factor', [None,
#                                             sample1(50), # should match p=50 below
#                                             sample2(50)])
# @pytest.mark.parametrize('sample_weight', [None, np.ones, sample1, sample2])
# def test_limits(limits,
#                 penalty_factor,
#                 sample_weight,
#                 df_max=None,
#                 covariance=None,
#                 standardize=True,
#                 fit_intercept=True,
#                 exclude=[],
#                 nlambda=None,
#                 lambda_min_ratio=None,
#                 n=100,
#                 p=50):

#     lower_limits, upper_limits = limits
#     if lower_limits is not None:
#         lower_limits = np.ones(p) * lower_limits
#     if upper_limits is not None:
#         upper_limits = np.ones(p) * upper_limits
#     X = rng.standard_normal((n, p))
#     Y = rng.standard_normal(n)
#     L = GaussNet(covariance=covariance,
#                  standardize=standardize,
#                  fit_intercept=fit_intercept,
#                  lambda_min_ratio=lambda_min_ratio,
#                  upper_limits=upper_limits,
#                  lower_limits=lower_limits,
#                  penalty_factor=penalty_factor,
#                  df_max=df_max)
#     if nlambda is not None:
#         L.nlambda = nlambda

#     if sample_weight is not None:
#         weights = sample_weight(n)
#     else:
#         weights = None

#     L.fit(X,
#           Y,
#           exclude=exclude,
#           sample_weight=weights)

#     C = get_glmnet_soln(X,
#                         Y,
#                         covariance=covariance,
#                         standardize=standardize,
#                         fit_intercept=fit_intercept,
#                         lower_limits=lower_limits,
#                         upper_limits=upper_limits,
#                         penalty_factor=penalty_factor,
#                         exclude=exclude,
#                         weights=weights,
#                         nlambda=nlambda,
#                         df_max=df_max,
#                         lambda_min_ratio=lambda_min_ratio)

#     if np.any(lower_limits == 0) or np.any(upper_limits == 0):
#         tol = 1e-3
#     else:
#         tol = 1e-10
#     print(tol, 'tol')
#     assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
#     if fit_intercept:
#         assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

@pytest.mark.parametrize('offset', [None, np.zeros(100), 20*sample1(100)]) # should match n=100 below
@pytest.mark.parametrize('penalty_factor', [None,
                                            sample1(50), # should match p=50 below
                                            sample2(50)])
@pytest.mark.parametrize('sample_weight', [None, sample1, sample2])
def test_offset(offset,
                penalty_factor,
                sample_weight,
                df_max=None,
                covariance=None,
                standardize=True,
                fit_intercept=True,
                exclude=[],
                nlambda=None,
                lambda_min_ratio=None,
                n=100,
                p=50):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n) * 100
    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 df_max=df_max)
    if nlambda is not None:
        L.nlambda = nlambda

    if sample_weight is not None:
        weights = sample_weight(n)
    else:
        weights = None

    L.fit(X,
          Y,
          exclude=exclude,
          sample_weight=weights,
          offset=offset)

    C = get_glmnet_soln(X,
                        Y.copy(),
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weights,
                        nlambda=nlambda,
                        offset=offset,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    tol = 1e-10
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol


