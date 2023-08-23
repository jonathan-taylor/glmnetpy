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

    
    
    W = rng.uniform(0, 1, size=(n,))
    design = _get_design(X, W)
    coef_ = rng.standard_normal(p)
    int_ = rng.standard_normal()

    beta = np.zeros(p+1)
    beta[0] = int_
    beta[1:] = coef_
    
    state = GLMNetState(coef_, int_)
    state.update(design, GLM.family, GLM.offset)

    eta = X1 @ beta
    mu = np.exp(eta) / (1 + np.exp(eta))

    Z = eta + (y - state.mu) / (state.mu * (1 - state.mu))
    print(Z[:3], 'Zq')
    assert np.allclose(state.mu, mu)
    assert np.allclose(state.eta, eta)
    
    W_ = W * (state.mu * (1 - state.mu))
    G = X1.T @ (W_ * Z)
    H = X1.T @ (W_[:, None] * X1)
    
    print(W_[:5], 'Wq')
    new_beta = np.linalg.inv(H) @ G
    R = _quasi_newton_step(GLM,
                       design,
                       y,
                       W,
                       state)
    print(R[0].coef, R[0].intercept, 'quasi')
    print(new_beta, 'new')
    sm_G = sm.GLM(y, X, family=F)
    sm_G.fit()
    


# In[3]:


test_quasi_newton()


# In[ ]:




