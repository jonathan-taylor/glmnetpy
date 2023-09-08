from itertools import product

import numpy as np
import scipy.sparse

from glmnet.elnet import ElNet

def test_small_penalty(n=50, p=10):

    rng = np.random.default_rng(0)
    X  = rng.standard_normal((n, p))
    X_s = scipy.sparse.csc_array(X)

    y = rng.standard_normal(n)
    W = rng.uniform(0, 1, size=(n,))

    for intercept, standardize in product([False, True],
                                          [False, True]):
        spec = ElNet(lambda_val=0.5,
                     standardize=standardize,
                     fit_intercept=intercept)

        out = spec.fit(X, y, sample_weight=W).result_
        out_s = spec.fit(X_s, y, sample_weight=W).result_

        beta = out.beta.toarray().reshape(-1)
        beta_s = out_s.beta.toarray().reshape(-1)

        assert np.allclose(beta, beta_s, rtol=1e-3, atol=1e-3)
