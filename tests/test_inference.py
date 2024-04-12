import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.inference import (fixed_lambda_estimator,
                              lasso_inference)
@pytest.mark.parametrize('n', [500])
@pytest.mark.parametrize('p', [103])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('standardize', [True, False])
def test_inference(n,
                   p,
                   fit_intercept,
                   standardize,
                   ntrial=10):
    # run a few times to be sure KKT conditions not violated

    for _ in range(ntrial):
        run_inference(n,
                      p,
                      fit_intercept,
                      standardize)

def run_inference(n,
                  p,
                  fit_intercept,
                  standardize,
                  rng=None,
                  alt=False,
                  s=3,
                  prop=0.8):

    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    D = np.linspace(1, p, p) / p + 1
    X *= D[None,:]
    Y = rng.standard_normal(n) * 2
    beta = np.zeros(p)
    if alt:
        beta[:s] = rng.standard_normal(s)
    Y += X @ beta
    Df = pd.DataFrame({'response':Y})

    fam = sm.families.Gaussian()
    GN = GLMNet(response_id='response',
                family=fam,
                fit_intercept=fit_intercept,
                standardize=standardize)
    GN.fit(X, Df)
    m = int(prop*n)

    df = lasso_inference(GN, 
                         GN.lambda_values_[10],
                         (X[:m], Df.iloc[:m], None),
                         (X, Df, None))
    if fit_intercept:
        active_set = np.array(df.index[1:]).astype(int)
    else:
        active_set = np.array(df.index).astype(int)
    X_sel = X[:,active_set]

    if fit_intercept:
        X_sel = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
    targets = np.linalg.pinv(X_sel) @ X @ beta

    df['target'] = targets
    return df


def main_null(fit_intercept=True,
              standardize=True,
              n=2000,
              p=50,
              ntrial=500,
              rng=None):

    ncover = 0
    nsel = 0

    dfs = []

    rng = np.random.default_rng(0)
    for i in range(ntrial):
        df = run_inference(n,
                           p,
                           fit_intercept,
                           standardize,
                           rng=rng)
        dfs.append(df)
        all_df = pd.concat(dfs)
        ncover += ((all_df['lower'] < 0) & (all_df['upper'] > 0)).sum()
        nsel += all_df.shape[0]

        print('cover:', ncover / nsel, 'typeI:', (all_df['pval'] < 0.05).mean())


def main_alt(fit_intercept=True,
              standardize=True,
              n=2000,
              p=50,
              ntrial=500,
              rng=None):

    ncover = 0
    nsel = 0

    dfs = []

    rng = np.random.default_rng(0)
    for i in range(ntrial):
        df = run_inference(n,
                           p,
                           fit_intercept,
                           standardize,
                           rng=rng,
                           alt=True)
        dfs.append(df)
        all_df = pd.concat(dfs)
        ncover += ((all_df['lower'] < all_df['target']) & (all_df['upper'] > all_df['target'])).sum()
        nsel += all_df.shape[0]

        print('cover:', ncover / nsel)
    return all_df

        
