import pytest
import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.base import Design
rng = np.random.default_rng(0)

n, p, q = 100, 5, 3

@pytest.mark.parametrize('X', [rng.standard_normal((n, p)),
                               scipy.sparse.csc_array(rng.standard_normal((n, p)))] )
@pytest.mark.parametrize('weights', [np.ones(n), rng.uniform(1, 2, size=(n,))])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('intercept', [True, False])
def test_design(X, weights, standardize, intercept):
    """
    test the linear / adjoint maps of Design from an np.ndarray and a scipy.sparse.csc_array
    """

    #X = X.copy()

    n, p = X.shape
    W = weights
    
    X_s = scipy.sparse.csc_array(X)
    design = Design(X,
                    W,
                    standardize=standardize,
                    intercept=intercept)

    design_s = Design(X_s, W,
                      standardize=standardize,
                      intercept=intercept)

    if scipy.sparse.issparse(X):
        X_np = X.toarray()
    else:
        X_np = X.copy()

    xm = (X_np.T @ W / W.sum()).reshape(-1)
    xm2 = (X_np**2).T @ W / W.sum()
    xs = np.sqrt(xm2 - xm**2).reshape(-1)

    if standardize:
        X_np /= xs[None,:]
        xm = (X_np.T @ W / W.sum()).reshape(-1)
        xm2 = (X_np**2).T @ W / W.sum()
        xs = np.sqrt(xm2 - xm**2).reshape(-1)
        if intercept:
            X_np -= (xm/xs)[None,:]  
    else:
        xs = np.ones(p)
        if intercept:
            xm = (X_np.T @ W / W.sum()).reshape(-1)
            X_np -= xm[None,:]
            
    if intercept:
        assert(np.allclose(X_np.T @ W, 0))

    print('intercept:', intercept, 'standardize:', standardize)

    X_eff = np.concatenate([np.ones((n, 1)), X_np], axis=1)

    V_r = rng.standard_normal((p+1, q))
    V_l = rng.standard_normal((n, q))

    # multiply a vector

    assert np.allclose(X_eff @ V_r[:,0], design @ V_r[:,0])
    assert np.allclose(design @ V_r[:,0], design_s @ V_r[:,0])
    
    # multiply a matrix

    assert np.allclose(X_eff @ V_r, design @ V_r)
    assert np.allclose(design @ V_r, design_s @ V_r)
    
    # adjoint multiply a vector
    
    assert np.allclose(X_eff.T @ V_l[:,0], design.T @ V_l[:,0])
    assert np.allclose(design.T @ V_l[:,0], design_s.T @ V_l[:,0])
    
    # adjoint multiply a matrix

    assert np.allclose(X_eff.T @ V_l, design.T @ V_l)
    assert np.allclose(design.T @ V_l, design_s.T @ V_l)
    

# @pytest.mark.parametrize('X', [rng.standard_normal((n, p)),
#                                scipy.sparse.csc_array(rng.standard_normal((n, p)))] )
# @pytest.mark.parametrize('weights', [np.ones(n), rng.uniform(1, 2, size=(n,))])
# @pytest.mark.parametrize('standardize', [True, False])
# @pytest.mark.parametrize('intercept', [True, False])
# @pytest.mark.parametrize('gls', [None,
#                                  rng.uniform(1, 2, size=(n,)),
#                                  rng.uniform(1, 2, size=(n,n))])
# def test_quadratic_form(X, weights, standardize, intercept, gls):
#     """
#     test the linear / adjoint maps of Design from an np.ndarray and a scipy.sparse.csc_array
#     """
    
#     if gls is not None and gls.ndim == 2:
#         gls = (gls + gls.T) / 2
#     X_s = scipy.sparse.csc_array(X)
#     design = Design(X, W, standardize=stand)
#     design_s = Design(X_s, W, standardize=stand)

#     Q_s = design_s.quadratic_form(G=gls, columns=columns)
#     Q = design.quadratic_form(G=gls, columns=columns)

#     X_eff = scipy.sparse.csc_array(X).toarray()
#     if stand:
#         xm = (X_eff * W[:,None]).sum(0) / W.sum()
#         x2 = (X_eff**2 * W[:,None]).sum(0) / W.sum()
#         xs = np.sqrt(x2 - xm**2)
#     else:
#         xm = np.zeros(p)
#         xs = np.ones(p)

#     X_eff = X_eff / xs[None,:] - np.multiply.outer(np.ones(n), xm / xs)
#     X_eff = np.concatenate([np.ones((n,1)), X_eff], axis=1)

#     if columns is not None:
#         columns = np.array(columns)
#         columns += 1
#         columns = np.hstack([[0], columns])

#     if gls is None:
#         Q_eff = X_eff.T @ X_eff
#     elif gls.ndim == 1:
#         Q_eff = X_eff.T @ (gls[:, None] * X_eff)
#     else:
#         Q_eff = X_eff.T @ gls @ X_eff
#     if columns is not None:
#         Q_eff = Q_eff[:, columns]
#     assert np.allclose(Q, Q_eff)  # calculation is done correctly

#     assert np.allclose(Q_s, Q) # sparse and dense agree

        
