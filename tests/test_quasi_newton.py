from itertools import product
from dataclasses import asdict

import numpy as np
import scipy.sparse
import statsmodels.api as sm

from glmnet.glm import GLM, GLMState
from glmnet.irls import quasi_newton_step, IRLS
from glmnet.base import Design
from glmnet.elnet import ElNet
from glmnet._utils import _obj_function, _parent_dataclass_from_child

def test_quasi_newton(n=100, p=5):

    rng = np.random.default_rng(0)

    y = rng.standard_normal(n) > 0
    F = sm.families.Binomial()
    
    Xv = rng.standard_normal((n, p))
    WU = rng.uniform(0, 1, size=(n,))
    for W, s in product([WU, np.ones(n)],
                        [False,True]):

        glm = GLM(0, family=F, standardize=s)
        glm.vp = np.ones(p) # this would usually be set within `fit`
        elnet_est = _parent_dataclass_from_child(ElNet,
                                                 asdict(glm),
                                                 standardize=False)
        elnet_solver = elnet_est.fit
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

        state = GLMState(coef_, int_)
        state.update(design,
                     glm.family,
                     glm.offset)

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

        offset = None

        objective = partial(objective, y.copy(), W.copy(), glm.family, glm.regularizer)

        new_beta = beta + np.linalg.inv(H) @ G
        R, _, _, halved = quasi_newton_step(glm.regularizer,
                                            glm.family,
                                            design,
                                            y,
                                            offset,
                                            W,
                                            state,
                                            objective,
                                            glm.control)

        R_s, _, _, halved = quasi_newton_step(glm.regularizer,
                                              glm.family,
                                              design_s,
                                              y,
                                              offset,
                                              W,
                                              state,
                                              objective,
                                              glm.control)

        new_state = GLMState(coef_, int_)
        new_state.update(design,
                         glm.family,
                         glm.offset)

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

        GLM = GLM(0, family=F, standardize=s)
        elnet_est = _parent_dataclass_from_child(ElNet,
                                                 asdict(GLM),
                                                 standardize=False)
        elnet_solver = elnet_est.fit
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

        state = GLMState(coef_, int_)
        state.update(design,
                     GLM.family,
                     GLM.offset)

        _, _, _, glm_state = IRLS(GLM,
                                  design,
                                  y,
                                  W,
                                  state,
                                  elnet_solver)

        X_1 = np.concatenate([np.ones((n,1)), X], axis=1)
        res = sm.GLM(y, X_1, family=F, var_weights=W).fit()

        sm_state = GLMState(res.params[1:], res.params[0])
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

def test_GLM(n=100, p=5):

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

        glm = GLM(0, family=F, standardize=s)
        X = Xv
        design = Design(X, W, standardize=s)
        GLM.fit(X, y, weights=W)
        res = GLM.result_

        X_1 = np.concatenate([np.ones((n,1)), X], axis=1)
        res = sm.GLM(y, X_1, family=F, var_weights=W).fit()

        sm_state = GLMState(res.params[1:], res.params[0])
        sm_state.update(design,
                        glm.family,
                        glm.offset)
        print(_obj_function(y,
                            sm_state.mu,
                            W,
                            glm.family,
                            glm.lambda_val,
                            glm.alpha,
                            sm_state.coef,
                            glm.vp), 'sm_val')

        glm_state = GLMState(glm.coef_, glm.intercept_)
        glm_state.update(design,
                         glm.family,
                         glm.offset)

        print(_obj_function(y,
                            glm_state.mu,
                            W,
                            glm.family,
                            glm.lambda_val,
                            glm.alpha,
                            glm_state.coef,
                            glm.vp), 'glm_val')

        assert np.allclose(glm_state.coef, res.params[1:], rtol=1e-3, atol=1e-3)
        assert np.allclose(glm_state.intercept, res.params[0], rtol=1e-3, atol=1e-3)
        
def objective(y, normed_sample_weight, family, regularizer, state):
    val1 = family.deviance(y, state.mu, freq_weights=normed_sample_weight) / 2
    val2 = regularizer.objective(state)
    val = val1 + val2
    return val
