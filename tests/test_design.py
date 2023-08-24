from itertools import product

import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.base import Design

def test_design(n=100, p=5, q=3):
    """
    test the linear / adjoint maps of Design from an np.ndarray and a scipy.sparse.csc_array
    """
    rng = np.random.default_rng(0)
    
    for X, W, s in product([rng.standard_normal((n, p)),
                            scipy.sparse.csc_array(rng.standard_normal((n, p)))],
                           [rng.uniform(0, 1, size=(n,)), np.ones(n)],
                           [False, True]):

        X_s = scipy.sparse.csc_array(X)
        design = Design(X, W, standardize=s)
        design_s = Design(X_s, W, standardize=s)
        X_ = design.linear_map(np.identity(p))
        X_T = X_.T
        
        v_r = rng.standard_normal(p)
        v_l = rng.standard_normal(n)

        V_r = rng.standard_normal((p, q))
        V_l = rng.standard_normal((n, q))

        assert np.allclose(X_ @ v_r, design.linear_map(v_r, 0))
        assert np.allclose(X_ @ v_r + 2, design.linear_map(v_r, 2))
        assert np.allclose(X_ @ v_r, design_s.linear_map(v_r, 0))
        assert np.allclose(X_ @ v_r + 2, design_s.linear_map(v_r, 2))

        assert np.allclose(X_ @ V_r, design.linear_map(V_r, 0))
        assert np.allclose(X_ @ V_r + 2, design.linear_map(V_r, 2))
        assert np.allclose(X_ @ V_r, design_s.linear_map(V_r, 0))
        assert np.allclose(X_ @ V_r + 2, design_s.linear_map(V_r, 2))
        
        U1, U2 = design.adjoint_map(v_l)
        V1, V2 = design_s.adjoint_map(v_l)
        assert np.allclose(U1, X_T @ v_l)
        assert np.allclose(U2, v_l.sum(0))
        
        U1, U2 = design.adjoint_map(V_l)
        V1, V2 = design_s.adjoint_map(V_l)
        assert np.allclose(U1, X_T @ V_l)
        assert np.allclose(U2, V_l.sum(0))
        assert np.allclose(V1, U1)
        assert np.allclose(V2, U2)
        
        xm, xs = design.xm, design.xs
        if scipy.sparse.issparse(X):
            X_v = X.toarray()
        else:
            X_v = X
        X_v = X_v - np.multiply.outer(np.ones(n), xm)
        X_v = X_v / xs[None,:]

        assert np.allclose(X_, X_v)

        Q_s = design_s.quadratic_form(G=G)
        Q = design.quadratic_form(G=G)

        assert np.allclose(Q_s, Q)

def test_quadratic_form(n=100, p=5, q=3):
    """
    test the linear / adjoint maps of Design from an np.ndarray and a scipy.sparse.csc_array
    """
    rng = np.random.default_rng(0)
    
    for X, W, stand, G in product([rng.standard_normal((n, p)),
                               scipy.sparse.csc_array(rng.standard_normal((n, p)))],
                              [rng.uniform(0, 1, size=(n,)), np.ones(n)],
                              [False, True],
                              [None,
                               rng.uniform(0, 1, size=(n,)),
                               rng.uniform(0, 1, size=(n,n))]):

        if G is not None and G.ndim == 2:
            G = (G + G.T) / 2
        X_s = scipy.sparse.csc_array(X)
        design = Design(X, W, standardize=stand)
        design_s = Design(X_s, W, standardize=stand)

        Q_s = design_s.quadratic_form(G=G)
        Q = design.quadratic_form(G=G)

        X_eff = scipy.sparse.csc_array(X).toarray()
        if stand:
            xm = (X_eff * W[:,None]).sum(0) / W.sum()
            x2 = (X_eff**2 * W[:,None]).sum(0) / W.sum()
            xs = np.sqrt(x2 - xm**2)
        else:
            xm = np.zeros(p)
            xs = np.ones(p)

        X_eff = X_eff / xs[None,:] - np.multiply.outer(np.ones(n), xm / xs)
        X_eff = np.concatenate([np.ones((n,1)), X_eff], axis=1)

        if G is None:
            Q_ = X_eff.T @ X_eff
        elif G.ndim == 1:
            Q_ = X_eff.T @ (G[:, None] * X_eff)
        else:
            Q_ = X_eff.T @ G @ X_eff
        assert np.allclose(Q, Q_)  # calculation is done correctly

        # assert np.allclose(Q_s, Q) # sparse and dense agree

        
