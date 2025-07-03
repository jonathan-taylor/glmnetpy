import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from glmnet import MultiClassNet

rng = np.random.default_rng(0)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_dense_vs_sparse_multiclassnet(standardize, fit_intercept, n=90, p=7, K=4):
    """
    Compare MultiClassNet fits for dense vs. sparse X.
    """
    X = rng.standard_normal((n, p))
    y = rng.integers(0, K, size=n)
    df = pd.DataFrame({'response': y})
    # Dense fit
    net_dense = MultiClassNet(response_id='response',
                              fit_intercept=fit_intercept,
                              standardize=standardize)
    net_dense.fit(X, df)
    # Sparse fit
    X_sparse = scipy.sparse.csc_array(X)
    net_sparse = MultiClassNet(response_id='response',
                               fit_intercept=fit_intercept,
                               standardize=standardize)
    net_sparse.fit(X_sparse, df)
    # Compare paths
    for coef_dense, coef_sparse in zip(net_dense.coefs_, net_sparse.coefs_):
        assert np.allclose(coef_dense, coef_sparse, atol=1e-5, rtol=1e-4)
    assert np.allclose(net_dense.intercepts_, net_sparse.intercepts_, atol=1e-5) 