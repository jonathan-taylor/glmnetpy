from dataclasses import asdict

import pytest

import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.elnet import ElNetEstimator
from glmnet.glmnet import GLMNetEstimator

rng = np.random.default_rng(0)
n, p = 30, 10

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('lambda_val', [0, np.sqrt(n)])
@pytest.mark.parametrize('alpha', [0, 0.5, 1])
def test_compare_glmnet_elnet(standardize,
                              fit_intercept,
                              sample_weight,
                              lambda_val,
                              alpha):

    sample_weight = sample_weight(n)
    X = rng.normal(size=(n,p))
    coefs = np.zeros(p)
    coefs[[2,3]] = 3 / np.sqrt(n)
    y = rng.normal(size=n) + 1 + X @ coefs

    elnet = ElNetEstimator(lambda_val,
                           alpha=alpha,
                           standardize=standardize,
                           fit_intercept=fit_intercept)
    elnet.fit(X, 
              y,
              sample_weight)

    glmnet = GLMNetEstimator(lambda_val,
                             alpha=alpha,
                             standardize=standardize,
                             fit_intercept=fit_intercept)
    print('glmnet alpha', glmnet.alpha)
    glmnet.fit(X, 
               y,
               sample_weight)

    if not np.allclose(elnet.intercept_, glmnet.intercept_, rtol=1e-4):
        raise ValueError('intercepts not close')
    if not np.linalg.norm(elnet.coef_ - glmnet.coef_) / np.linalg.norm(glmnet.coef_) < 1e-4:
        raise ValueError('coefs not close')

    elnet_dict = asdict(elnet)
    glmnet_dict = asdict(glmnet)

    eta = X @ glmnet.coef_ + glmnet.intercept_
    dev = np.sum(sample_weight * (y - eta)**2) 
    # dev_ratio = 1 - dev / glmnet_dict['nulldev']

    # # check the regularizer 
    # assert(dev_ratio == glmnet_dict['dev_ratio'])
    
    # del(elnet_dict['a0']) # compared above
    # del(elnet_dict['beta']) # compared above
    # del(elnet_dict['warm_fit']) # don't compare
    # del(elnet_dict['npasses'])
    
    # failures = []

    # for k in elnet_dict:
    #     try:
    #         elnet_v = np.asarray(elnet_dict[k])
    #         glmnet_v = np.asarray(glmnet_dict[k])
    #     except:
    #         elnet_v = glmnet_v = 0
    #     if not np.allclose(elnet_v, glmnet_v):
    #         failures.append(f'field {k} differs {elnet_v}, {glmnet_v}')

    # if failures:
    #     raise ValueError(';'.join(failures))

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('lambda_val', [0, np.sqrt(n)])
def test_compare_sparse_elnet(standardize,
                              fit_intercept,
                              sample_weight,
                              lambda_val):

    sample_weight = sample_weight(n)
    X = rng.normal(size=(n,p))
    y = rng.normal(size=n)

    elnet = ElNetEstimator(lambda_val,
                           standardize=standardize,
                           fit_intercept=fit_intercept)
    elnet.fit(X, 
              y,
              sample_weight)

    Xs = scipy.sparse.csc_array(X).tocsc()
    elnet_s = ElNetEstimator(lambda_val,
                             standardize=standardize,
                             fit_intercept=fit_intercept)
    elnet_s.fit(Xs, 
                y,
                sample_weight)

    if not np.allclose(elnet.intercept_, elnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(elnet.coef_, elnet_s.coef_):
        raise ValueError('coefs not close')

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('lambda_val', [0, np.sqrt(n)])
def test_compare_sparse_glmnet(standardize,
                               fit_intercept,
                               sample_weight,
                               lambda_val):

    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    beta = np.zeros(p)
    beta[:2] = [2,-2.3]
    y = rng.normal(size=n) + X @ beta
    sample_weight = sample_weight(n)

    glmnet = GLMNetEstimator(lambda_val,
                             standardize=standardize,
                             fit_intercept=fit_intercept)
    glmnet.fit(X,
               y,
               sample_weight=sample_weight)

    Xs = scipy.sparse.csc_array(X).tocsc()
    glmnet_s = GLMNetEstimator(lambda_val,
                               standardize=standardize,
                               fit_intercept=fit_intercept)
    glmnet_s.fit(Xs,
                 y,
                 sample_weight=sample_weight)

    if not np.allclose(glmnet.intercept_, glmnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(glmnet.coef_, glmnet_s.coef_):
        raise ValueError('coefs not close')
    
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('lambda_val', [0, 0.5 / np.sqrt(n)])
def test_logistic(standardize,
                  fit_intercept,
                  sample_weight,
                  lambda_val):

    n, p = 100, 10
    sample_weight = sample_weight(n)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    glmnet = GLMNetEstimator(lambda_val,
                             standardize=standardize,
                             fit_intercept=fit_intercept,
                             family=sm.families.Binomial())
    glmnet.fit(X, 
               y,
               weights)
    print(glmnet.coef_)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: rng.uniform(0, 1, size=(n,))])
@pytest.mark.parametrize('lambda_val', [0, np.sqrt(n)])
def test_probit(standardize,
                fit_intercept,
                sample_weight,
                lambda_val):

    n, p = 100, 10
    sample_weight = sample_weight(n)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    link = sm.families.links.Probit()
    glmnet = GLMNetEstimator(lambda_val,
                             standardize=standardize,
                             fit_intercept=fit_intercept,
                             family=sm.families.Binomial(link=link))
    glmnet.fit(X, 
               y,
               weights)
    print(glmnet.coef_)
