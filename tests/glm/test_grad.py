import numpy as np
import pytest

from glmnet.inference import GLMNetInference
from glmnet import (GLMNet,
                    compute_grad,
                    GLM)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('glmnet', [True, False])
def test_compute_grad(standardize,
                      fit_intercept,
                      glmnet):

    n, p, prop = 100, 50, 0.8
    scale = 2
    rng = np.random.default_rng(0)

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n) * scale

    lambda_val = 1e-3


    # In[ ]:

    if glmnet:
        G = GLMNet(lambda_values=[lambda_val],
                   fit_intercept=fit_intercept,
                   standardize=standardize)
        G.fit(X, Y)
    else:
        G = GLM(fit_intercept=fit_intercept,
                standardize=standardize)
        G.fit(X, Y)

    I = G.design_.unscaler_ @ G.design_.scaler_
    x = rng.standard_normal(I.shape[0])
    assert np.allclose(x, I @ x)
    
    coef = rng.standard_normal(p)
    intercept = rng.standard_normal()
    
    grad, resid = compute_grad(G,
                        intercept,
                        coef,
                        G.design_,
                        Y)

    fit = X @ coef + intercept
    grad_by_hand = X.T @ (Y - fit)

    assert np.allclose(resid, Y - fit)
    assert np.allclose(grad[1:], grad_by_hand)
    assert np.allclose(grad[0], np.sum(Y - fit))
