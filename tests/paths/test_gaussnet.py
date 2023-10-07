import numpy as np
import pandas as pd

import pytest
rng = np.random.default_rng(0)

from sklearn.model_selection import KFold

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
                    weights=None,
                    foldid=None,
                    alignment='lambda'):

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
            args.append('type.gaussian="covariance"')

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

        cvargs = ','.join([args, f'alignment="{alignment}"'])
        CV = True
        rpy.r.assign('doCV', foldid is not None)
        if foldid is not None:
            rpy.r.assign('foldid', foldid)
        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
library(glmnet)
Y = as.numeric(Y)
X = as.matrix(X)
G = glmnet(X, Y, {args})
C = as.matrix(coef(G))
if (doCV) {{
    foldid = as.integer(foldid)
    CVG = cv.glmnet(X, Y, {cvargs}, foldid=foldid, grouped=TRUE)
    CVM = CVG$cvm
    CVSD = CVG$cvsd
}}
            '''
        print(cmd)

        rpy.r(cmd)
        C = rpy.r('C')
        if foldid is not None:
            CVM = rpy.r('CVM')
            CVSD = rpy.r('CVSD')
    if foldid is None:
        return C.T
    else:
        return C.T, CVM, CVSD

# weight functions

def sample1(n):
    return rng.uniform(0, 1, size=n)

def sample2(n):
    V = sample1(n)
    V[:n//2] = 0
    return V

@pytest.mark.parametrize('sample_weight', [None, np.ones, sample1, sample2])
@pytest.mark.parametrize('df_max', [None, 5])
@pytest.mark.parametrize('exclude', [[], [1,2,3]])
@pytest.mark.parametrize('lower_limits', [-1, None])
# covariance changes type.gaussian, behaves unpredictably even in R
@pytest.mark.parametrize('covariance', [None]) 
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('nlambda', [None, 20])
@pytest.mark.parametrize('lambda_min_ratio', [None,0.02])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
def test_gaussnet(covariance,
                  standardize,
                  fit_intercept,
                  exclude,
                  lower_limits,
                  nlambda,
                  lambda_min_ratio,
                  sample_weight,
                  df_max,
                  n,
                  p):

    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame({'Y':Y})
    col_args = {'response_col':'Y'}
    
    if sample_weight is not None:
        weights = sample_weight(n)
        D['weights'] = weights
        col_args['weight_col'] = 'weights'
    else:
        weights = None

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 exclude=exclude,
                 df_max=df_max, **col_args)

    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        lower_limits=lower_limits,
                        exclude=exclude,
                        weights=weights,
                        nlambda=nlambda,
                        lambda_min_ratio=lambda_min_ratio,
                        df_max=df_max)


    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10


@pytest.mark.parametrize('limits', [(-1, np.inf), (-np.inf, 1),
                                    (-np.inf, 0), (0, np.inf),
                                    (-np.inf, np.inf),
                                    (-1, 1),
                                    (0, 1)])
@pytest.mark.parametrize('penalty_factor', [None,
                                            sample1(50), # should match p=50 below
                                            sample2(50)])
@pytest.mark.parametrize('sample_weight', [None, np.ones, sample1, sample2])
def test_limits(limits,
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

    lower_limits, upper_limits = limits
    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits
    if upper_limits is not None:
        upper_limits = np.ones(p) * upper_limits
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame({'Y':Y})
    col_args = {'response_col':'Y'}
    
    if sample_weight is not None:
        weights = sample_weight(n)
        D['weights'] = weights
        col_args['weight_col'] = 'weights'
    else:
        weights = None

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 upper_limits=upper_limits,
                 lower_limits=lower_limits,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weights,
                        nlambda=nlambda,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    if np.any(lower_limits == 0) or np.any(upper_limits == 0):
        tol = 1e-3
    else:
        tol = 1e-10
    print(tol, 'tol')
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

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
    D = pd.DataFrame({'Y':Y})
    col_args = {'response_col':'Y'}

    if sample_weight is not None:
        weights = sample_weight(n)
        D['weights'] = weights
        col_args['weight_col'] = 'weights'
    else:
        weights = None

    if offset is not None:
        D['offset'] = offset
        col_args['offset_col'] = 'offset'

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

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

@pytest.mark.parametrize('offset', [None, np.zeros(103), 20*sample1(103)]) # should match n=103 below
@pytest.mark.parametrize('penalty_factor', [None,
                                            sample1(50), # should match p=50 below
                                            sample2(50)])
@pytest.mark.parametrize('sample_weight', [None, sample1])
@pytest.mark.parametrize('alignment', ['lambda', 'fraction'])
def test_CV(offset,
            penalty_factor,
            sample_weight,
            alignment,
            df_max=None,
            covariance=None,
            standardize=True,
            fit_intercept=True,
            exclude=[],
            nlambda=None,
            lambda_min_ratio=None,
            n=103,
            p=50):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n) * 100
    D = pd.DataFrame({'Y':Y})
    col_args = {'response_col':'Y'}

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    if sample_weight is not None:
        weights = sample_weight(n)
        D['weights'] = weights
        col_args['weight_col'] = 'weights'
    else:
        weights = None

    if offset is not None:
        D['offset'] = offset
        col_args['offset_col'] = 'offset'

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)
    L.cross_validation_path(X,
                            D,
                            alignment=alignment,
                            cv=cv)
    CVM_ = L.cv_scores_['Mean Squared Error']
    CVSD_ = L.cv_scores_['SD(Mean Squared Error)']
    C, CVM, CVSD = get_glmnet_soln(X,
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
                                   lambda_min_ratio=lambda_min_ratio,
                                   foldid=foldid,
                                   alignment=alignment)

    print(CVM, CVM_)
    assert np.allclose(CVM, CVM_)
    assert np.allclose(CVSD, CVSD_)

