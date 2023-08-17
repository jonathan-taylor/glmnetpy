import numpy as np
import scipy.sparse
from glmnet.elnet_fit import elnet_fit
from glmnet.glmnet_fit import glmnet_fit

def test_compare_glmnet_elnet():

    n, p = 30, 10
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n,p))
    y = rng.normal(size=n)
    lambda_val = 0.5 / np.sqrt(n)
    weights = np.ones(n) + rng.uniform(0, 1, size=(n,))
    weights /= weights.sum()

    elnet = elnet_fit(X, 
                      y,
                      weights,
                      lambda_val)

    glmnet = glmnet_fit(X, 
                        y,
                        weights,
                        lambda_val)

    if not np.allclose(elnet.a0, glmnet.a0):
        raise ValueError('intercepts not close')
    if not np.allclose(elnet.beta.toarray(), glmnet.beta.toarray()):
        raise ValueError('coefs not close')

    elnet_dict = elnet.__dict__
    glmnet_dict = glmnet.__dict__

    del(elnet_dict['beta']) # sparse, compared above
    del(elnet_dict['warm_fit']) # dict
    
    for k in elnet_dict:
        try:
            elnet_v = np.asarray(elnet_dict[k])
            glmnet_v = np.asarray(glmnet_dict[k])
            print(k, elnet_v, glmnet_v)
        except:
            elnet_v = glmnet_v = 0
        if not np.allclose(elnet_v, glmnet_v):
            print(f'field {k} differs: {elnet_v}, {glmnet_v}')
            #raise ValueError(f'field {k} differs')

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
    
