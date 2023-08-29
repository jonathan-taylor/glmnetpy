from dataclasses import asdict

import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.elnet import ElNetEstimator
from glmnet.glmnet import GLMNetEstimator

def test_compare_glmnet_elnet():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    coefs = np.zeros(p)
    coefs[[2,3]] = 3 / np.sqrt(n)
    y = rng.normal(size=n) + 1 + X @ coefs
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))

    elnet = ElNetEstimator(lambda_val)
    elnet.fit(X, 
              y,
              weights)

    glmnet = GLMNetEstimator(lambda_val)
    glmnet.fit(X, 
               y,
               weights)

    if not np.allclose(elnet.intercept_, glmnet.intercept_, rtol=1e-4):
        raise ValueError('intercepts not close')
    if not np.linalg.norm(elnet.coef_ - glmnet.coef_) / np.linalg.norm(glmnet.coef_) < 1e-4:
        raise ValueError('coefs not close')

    elnet_dict = asdict(elnet)
    glmnet_dict = asdict(glmnet)

    eta = X @ glmnet.coef_ + glmnet.intercept_
    dev = np.sum(weights * (y - eta)**2) 
    dev_ratio = 1 - dev / glmnet_dict['nulldev']

    # check the regularizer 
    assert(dev_ratio == glmnet_dict['dev_ratio'])
    
    del(elnet_dict['a0']) # compared above
    del(elnet_dict['beta']) # compared above
    del(elnet_dict['warm_fit']) # don't compare
    del(elnet_dict['npasses'])
    
    failures = []

    for k in elnet_dict:
        try:
            elnet_v = np.asarray(elnet_dict[k])
            glmnet_v = np.asarray(glmnet_dict[k])
        except:
            elnet_v = glmnet_v = 0
        if not np.allclose(elnet_v, glmnet_v):
            failures.append(f'field {k} differs {elnet_v}, {glmnet_v}')

    if failures:
        raise ValueError(';'.join(failures))

def test_compare_sparse_elnet():

    n, p = 30, 22
    rng = np.random.default_rng(0)


    X = rng.normal(size=(n,p))
    y = rng.normal(size=n)
    lambda_val = 2 * np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))

    elnet = ElNetEstimator(lambda_val)
    elnet.fit(X, 
              y,
              weights)

    Xs = scipy.sparse.csc_array(X).tocsc()
    elnet_s = ElNetEstimator(lambda_val)
    elnet_s.fit(Xs, 
                y,
                weights)

    if not np.allclose(elnet.intercept_, elnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(elnet.coef_, elnet_s.coef_):
        raise ValueError('coefs not close')

def test_compare_sparse_glmnet():

    n, p = 300, 22
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    beta = np.zeros(p)
    beta[:2] = [2,-2.3]
    y = rng.normal(size=n) + X @ beta
    lambda_val = 2 * np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights *= n / weights.sum()

    glmnet = GLMNetEstimator(lambda_val)
    glmnet.fit(X.copy(), 
               y,
               sample_weight=weights)

    Xs = scipy.sparse.csc_array(X.copy()).tocsc()
    glmnet_s = GLMNetEstimator(lambda_val)
    glmnet_s.fit(Xs, 
                 y,
                 sample_weight=weights)

    if not np.allclose(glmnet.intercept_, glmnet_s.intercept_):
        raise ValueError('intercepts not close')
    if not np.allclose(glmnet.coef_, glmnet_s.coef_):
        raise ValueError('coefs not close')
    
def test_logistic():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    glmnet = GLMNetEstimator(lambda_val, family=sm.families.Binomial())
    glmnet.fit(X, 
               y,
               weights,
               )
    print(glmnet.coef_)

def test_probit():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    link = sm.families.links.Probit()
    glmnet = GLMNetEstimator(lambda_val, family=sm.families.Binomial(link=link))

    glmnet.fit(X, 
               y,
               weights)
    print(glmnet.coef_)
    
