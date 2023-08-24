#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import statsmodels.api as sm

from glmnet.glmnet_fit import GLMNetEstimator, _quasi_newton_step, GLMNetState
from glmnet.glmnet_fit import _get_design



# In[2]:


def test_quasi_newton(n=100, p=5):

    rng = np.random.default_rng(0)

    X = rng.standard_normal((n, p))
    X1 = np.concatenate([np.ones((n,1)), X], axis=1)
    
    y = rng.standard_normal(n) > 0
    F = sm.families.Binomial()
    GLM = GLMNetEstimator(0, family=F)
    
    for W in [rng.uniform(0, 1, size=(n,)), np.ones(n)]:


        design = _get_design(X, W)
        coef_ = rng.standard_normal(p)
        int_ = rng.standard_normal()

        beta = np.zeros(p+1)
        beta[0] = int_
        beta[1:] = coef_

        state = GLMNetState(coef_, int_)
        state.update(design,
                     GLM.family,
                     GLM.offset)

        eta = X1 @ beta
        mu = np.exp(eta) / (1 + np.exp(eta))

        Z = eta + (y - state.mu) / (state.mu * (1 - state.mu))

        assert np.allclose(state.mu, mu)
        assert np.allclose(state.eta, eta)

        IRLS_W = W * (state.mu * (1 - state.mu))
        G = X1.T @ (W * (y - state.mu))
        G2 = X1.T @ (IRLS_W * (Z - eta))

        assert np.allclose(G, G2)

        H = X1.T @ (IRLS_W[:, None] * X1)

        new_beta = beta + np.linalg.inv(H) @ G
        R, _, _, halved = _quasi_newton_step(GLM,
                                             design,
                                             y,
                                             W,
                               state)

        assert not halved
        assert np.fabs((R.intercept - new_beta[0]) / new_beta[0]) < 1e-3
        assert np.allclose(R.coef, new_beta[1:], rtol=1e-3)

