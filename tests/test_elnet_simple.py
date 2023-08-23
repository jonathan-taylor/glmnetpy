import numpy as np

from glmnet.elnet_fit import ElNetEstimator

def test_simple(n=50, p=10):

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    W = rng.uniform(0, 1, size=(n,))

    spec = ElNetEstimator(lambda_val=0, weights=W)
    out = spec.fit(X, y, weights=W).result_

    beta = out.beta.toarray().reshape(-1)
    intercept = out.a0

    X1 = np.concatenate([np.ones((n,1)), X], axis=1)
    Q = X1.T @ (W[:, None] * X1)
    beta_wls = np.linalg.solve(Q, X1.T @ (W * y))

    assert np.allclose(beta_wls[1:], beta, rtol=1e-3, atol=1e-3)
    assert np.fabs((beta_wls[0] - intercept) / intercept) < 1e-3
