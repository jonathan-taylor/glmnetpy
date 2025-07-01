import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from glmnet import MultiGaussNet

rng = np.random.default_rng(0)

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('q', [2, 3])  # number of response variables
def test_dense_vs_sparse_path_multigaussnet(standardize, fit_intercept, q, n=100, p=10):
    """
    Compare path fits for dense vs. sparse X for MultiGaussNet.
    """
    X = rng.standard_normal((n, p))
    X[:, 4] *= 2.5
    X[:, 2] *= 1.2
    X[:, 0] *= 0.7
    X[:, 1] *= 0.9
    
    # Create coefficients for multiple responses
    beta = np.zeros((p, q))
    beta[:2, 0] = [1, 2]  # first response
    beta[:3, 1] = [0.5, 1.5, 0.8]  # second response
    if q > 2:
        beta[1:4, 2] = [0.3, 1.2, 0.6]  # third response
    
    # Generate multiple response variables
    Y = 2 * rng.standard_normal((n, q)) + X @ beta
    
    # Create DataFrame with multiple response columns
    response_cols = [f'response[{i}]' for i in range(q)]
    df_data = {col: Y[:, i] for i, col in enumerate(response_cols)}
    df = pd.DataFrame(df_data)
    
    # Dense fit
    net_dense = MultiGaussNet(response_id=response_cols,
                             fit_intercept=fit_intercept,
                             standardize=standardize)
    net_dense.fit(X, df)
    
    # Sparse fit
    X_sparse = scipy.sparse.csc_array(X)
    net_sparse = MultiGaussNet(response_id=response_cols,
                              fit_intercept=fit_intercept,
                              standardize=standardize)
    net_sparse.fit(X_sparse, df)
    
    # Compare paths (for each lambda)
    for coef_dense, coef_sparse in zip(net_dense.coefs_, net_sparse.coefs_):
        assert np.allclose(coef_dense, coef_sparse, atol=1e-5, rtol=1e-4)
    assert np.allclose(net_dense.intercepts_, net_sparse.intercepts_, atol=1e-5) 