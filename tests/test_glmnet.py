from dataclasses import asdict

import numpy as np
import scipy.sparse
from glmnet.elnet_fit import elnet_fit
from glmnet.glmnet_fit import glmnet_fit

def test_compare_glmnet_elnet():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    coefs = np.zeros(p)
    coefs[[2,3]] = 3 / np.sqrt(n)
    y = rng.normal(size=n) + 1 + X @ coefs
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))

    elnet = elnet_fit(X, 
                      y,
                      weights,
                      lambda_val)

    glmnet, spec = glmnet_fit(X, 
                              y,
                              weights,
                              lambda_val)

    if not np.allclose(elnet.a0, glmnet.a0, rtol=1e-4):
        raise ValueError('intercepts not close')
    if not np.linalg.norm(elnet.beta.toarray() - glmnet.beta.toarray()) / np.linalg.norm(glmnet.beta.toarray()) < 1e-4:
        raise ValueError('coefs not close')

    elnet_dict = asdict(elnet)
    glmnet_dict = asdict(glmnet)

    eta = X @ glmnet_dict['beta'].toarray().reshape(-1) + glmnet_dict['a0']
    dev = np.sum(weights * (y - eta)**2) 
    dev_ratio = 1 - dev / glmnet_dict['nulldev']
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

    elnet = elnet_fit(X, 
                      y,
                      weights,
                      lambda_val)

    Xs = scipy.sparse.csc_array(X).tocsc()
    elnet_s = elnet_fit(Xs, 
                        y,
                        weights,
                        lambda_val)

    if not np.allclose(elnet.a0, elnet_s.a0):
        raise ValueError('intercepts not close')
    if not np.allclose(elnet.beta.toarray(), elnet_s.beta.toarray()):
        raise ValueError('coefs not close')

def test_compare_sparse_glmnet():

    n, p = 30, 22
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.normal(size=n)
    lambda_val = 2 * np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))

    glmnet = glmnet_fit(X, 
                        y,
                        weights,
                        lambda_val)

    Xs = scipy.sparse.csc_array(X).tocsc()
    glmnet_s = glmnet_fit(Xs, 
                          y,
                          weights,
                          lambda_val)

    if not np.allclose(glmnet.a0, glmnet_s.a0):
        raise ValueError('intercepts not close')
    if not np.allclose(glmnet.beta.toarray(), glmnet_s.beta.toarray()):
        raise ValueError('coefs not close')
    
def test_logistic():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    glmnet = glmnet_fit(X, 
                        y,
                        weights,
                        lambda_val,
                        family='Binomial')
    print(glmnet.beta.toarray())

def test_probit():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.choice([0,1], size=n, replace=True)
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    glmnet = glmnet_fit(X, 
                        y,
                        weights,
                        lambda_val,
                        family='Binomial',
                        link='Probit')
    print(glmnet.beta.toarray())
    
