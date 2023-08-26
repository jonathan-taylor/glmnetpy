import numpy as np
import scipy.sparse

from glmnet.elnet import ElNetEstimator

def test_simple_ridge(n=50, p=10):

    rng = np.random.default_rng(0)
    X  = rng.standard_normal((n, p))
    X_s = scipy.sparse.csc_array(X)

    y = rng.standard_normal(n)
    W = rng.uniform(0, 1, size=(n,))

    for standardize in [False, True]:
        spec = ElNetEstimator(lambda_val=0, standardize=standardize)

        out = spec.fit(X, y, sample_weight=W).result_
        out_s = spec.fit(X_s, y, sample_weight=W).result_

        beta = out.beta.toarray().reshape(-1)
        beta_s = out_s.beta.toarray().reshape(-1)

        assert np.allclose(beta, beta_s, rtol=1e-3, atol=1e-3)
        intercept = out.a0

        if not standardize:
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)
        else:
            xm = (X * W[:,None]).sum(0) / W.sum()
            x2 = (X**2 * W[:,None]).sum(0) / W.sum()
            xs = np.sqrt(x2 - xm**2)
            X = (X - xm[None,:]) / xs
            X1 = np.concatenate([np.ones((n,1)), X], axis=1)

        Q = X1.T @ (W[:, None] * X1)
        beta_wls = np.linalg.solve(Q, X1.T @ (W * y))

        assert np.allclose(beta_wls[1:], beta, rtol=1e-3, atol=1e-3)
        assert np.fabs((beta_wls[0] - intercept) / intercept) < 1e-3
