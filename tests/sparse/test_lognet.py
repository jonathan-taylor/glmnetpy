import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from glmnet import LogNet

rng = np.random.default_rng(0)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_dense_vs_sparse_path_lognet(standardize, fit_intercept, n=100, p=10):
    """
    Compare path fits for dense vs. sparse X for LogNet.
    """
    X = rng.standard_normal((n, p))
    X[:, 4] *= 2.5
    X[:, 2] *= 1.2
    X[:, 0] *= 0.7
    X[:, 1] *= 0.9
    beta = np.zeros(p)
    beta[:2] = [1, 2]
    y = (X @ beta + rng.standard_normal(n) > 0).astype(int)
    df = pd.DataFrame({'response': y})
    # Dense fit
    net_dense = LogNet(response_id='response',
                      fit_intercept=fit_intercept,
                      standardize=standardize)
    net_dense.fit(X, df)
    # Sparse fit
    X_sparse = scipy.sparse.csc_array(X)
    net_sparse = LogNet(response_id='response',
                       fit_intercept=fit_intercept,
                       standardize=standardize)
    net_sparse.fit(X_sparse, df)
    # Compare paths (for each lambda)
    for coef_dense, coef_sparse in zip(net_dense.coefs_, net_sparse.coefs_):
        assert np.allclose(coef_dense, coef_sparse, atol=1e-5, rtol=1e-4)
    assert np.allclose(net_dense.intercepts_, net_sparse.intercepts_, atol=1e-5) 