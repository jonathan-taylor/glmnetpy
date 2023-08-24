from itertools import product

import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.glmnet_fit import GLMNetEstimator, _quasi_newton_step, _IRLS, GLMNetState
from glmnet.glmnet_fit import Design
from glmnet._utils import _obj_function

def test_quasi_newton(n=100, p=5):

    rng = np.random.default_rng(0)

    y = rng.standard_normal(n) > 0
    F = sm.families.Binomial()
    
    Xv = rng.standard_normal((n, p))
    WU = rng.uniform(0, 1, size=(n,))
    for W, s in product([WU, np.ones(n)],
                        [False,True]):

        GLM = GLMNetEstimator(0, family=F, standardize=s)
        X = Xv
        if not s:
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)
        else:
            xm = (X * W[:,None]).sum(0) / W.sum()
            x2 = (X**2 * W[:,None]).sum(0) / W.sum()
            xs = np.sqrt(x2 - xm**2)
            X = (X - xm[None,:]) / xs
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)
            
        Xs = scipy.sparse.csc_array(Xv)

        design = Design(X, W, standardize=s)
        design_s = Design(Xs, W, standardize=s)

        X_ = design.linear_map(np.identity(p), 0)
        coef_ = rng.standard_normal(p)
        int_ = rng.standard_normal()

        beta = np.zeros(p+1)
        beta[0] = int_
        beta[1:] = coef_

        state = GLMNetState(coef_, int_)
        state.update(design,
                     GLM.family,
                     GLM.offset)

        eta = design.linear_map(beta[1:], beta[0])
        
        mu = np.exp(eta) / (1 + np.exp(eta))

        Z = eta + (y - state.mu) / (state.mu * (1 - state.mu))

        assert np.allclose(state.mu, mu)
        assert np.allclose(state.eta, eta)
        assert np.allclose(eta, X_ @ beta[1:] + beta[0])
        
        IRLS_W = W * (state.mu * (1 - state.mu))
        G_coef, G_int = design_s.adjoint_map(W * (y - state.mu))
        G = np.zeros(p+1)
        G[0] = G_int
        G[1:] = G_coef
        H_c, H_1 = design_s.adjoint_map(IRLS_W[:, None] * X_)
        H = np.zeros((p+1,p+1))
        H[1:,1:] = H_c
        H[1:,0] = H[0,1:] = H_1
        H[0,0] = IRLS_W.sum()

        H_ = X1.T @ (IRLS_W[:,None] * X1)
        G_ = X1.T @ (W * (y - state.mu))
        assert np.allclose(H_, H)
        assert np.allclose(G_, G)
        assert np.allclose(H[1:,1:], X_.T @ (IRLS_W[:, None] * X_))
        assert np.allclose(H[0, 1:], (IRLS_W[:, None] * X_).sum(0))

        new_beta = beta + np.linalg.inv(H) @ G
        R, _, _, halved = _quasi_newton_step(GLM,
                                             design,
                                             y,
                                             W,
                                             state)

        R_s, _, _, halved = _quasi_newton_step(GLM,
                                               design_s,
                                               y,
                                               W,
                                               state)

        new_state = GLMNetState(coef_, int_)
        new_state.update(design,
                         GLM.family,
                         GLM.offset)

        # check that quasi newton gives the same whether sparse or not
        assert np.allclose(R_s.coef, R.coef)
        assert np.allclose(R_s.intercept, R.intercept)

        assert not halved
        assert np.fabs((R.intercept - new_beta[0]) / new_beta[0]) < 1e-3
        assert np.allclose(R.coef, new_beta[1:], rtol=1e-3)

        assert np.fabs((R_s.intercept - new_beta[0]) / new_beta[0]) < 1e-3
        assert np.allclose(R_s.coef, new_beta[1:], rtol=1e-3)
        
def test_IRLS(n=100, p=5):

    rng = np.random.default_rng(0)

    y = rng.standard_normal(n) > 0
    F = sm.families.Binomial()
    L = sm.families.links.Probit()
    F2 = sm.families.Binomial(link=L)
    
    Xv = rng.standard_normal((n, p))
    WU = rng.uniform(0, 1, size=(n,))

    for W, s, F in product([WU, np.ones(n)],
                           [False,True],
                           [F, F2]):

        GLM = GLMNetEstimator(0, family=F, standardize=s)
        GLM.vp = np.ones(p) # this would usually be set within `fit`
        X = Xv
        if not s:
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)
        else:
            xm = (X * W[:,None]).sum(0) / W.sum()
            x2 = (X**2 * W[:,None]).sum(0) / W.sum()
            xs = np.sqrt(x2 - xm**2)
            X = (X - xm[None,:]) / xs
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)
            
        Xs = scipy.sparse.csc_array(Xv)

        design = Design(X, W, standardize=s)
        design_s = Design(Xs, W, standardize=s)

        coef_ = rng.standard_normal(p) * 0.1
        int_ = rng.standard_normal() * 0.1

        beta = np.zeros(p+1)
        beta[0] = int_
        beta[1:] = coef_

        state = GLMNetState(coef_, int_)
        state.update(design,
                     GLM.family,
                     GLM.offset)

        _, _, _, glm_state = _IRLS(GLM,
                                   design,
                                   y,
                                   W,
                                   state)

        X_1 = np.concatenate([np.ones((n,1)), X], axis=1)
        res = sm.GLM(y, X_1, family=F, var_weights=W).fit()

        sm_state = GLMNetState(res.params[1:], res.params[0])
        sm_state.update(design,
                        GLM.family,
                        GLM.offset)
        print(_obj_function(y,
                            sm_state.mu,
                            W,
                            GLM.family,
                            GLM.lambda_val,
                            GLM.alpha,
                            sm_state.coef,
                            GLM.vp), 'sm_val')

        print(_obj_function(y,
                            glm_state.mu,
                            W,
                            GLM.family,
                            GLM.lambda_val,
                            GLM.alpha,
                            glm_state.coef,
                            GLM.vp), 'glm_val')

        assert np.allclose(glm_state.coef, res.params[1:], rtol=1e-3, atol=1e-3)
        assert np.allclose(glm_state.intercept, res.params[0], rtol=1e-3, atol=1e-3)

