import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.inference import lasso_inference

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('p', [103])
@pytest.mark.parametrize('n', [500])
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

def test_Auto():

    df = sm.datasets.get_rdataset('Auto', package='ISLR').data
    df = df.set_index('name')
    X = np.asarray(df.drop(columns=['mpg']))
    y = np.asarray(df['mpg'])

    GN = GLMNet(response_id='response',
                fit_intercept=True,
                standardize=True)

    Df = pd.DataFrame({'response':y})

    GN.fit(X, Df)
    prop = 0.8
    n =	df.shape[0]
    m = int(prop*n)

    df = lasso_inference(GN,
                         GN.lambda_values_[min(10, GN.lambda_values_.shape[0]-1)],
                         (X[:m], Df.iloc[:m], None),
                         (X, Df, None))
    print(df)


def run_inference(n,
                  p,
                  fit_intercept,
                  standardize,
                  rng=None,
                  alt=False,
                  s=3,
                  prop=0.75,
                  penalty_facs=True,
                  family='gaussian'):

    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    D = np.linspace(1, p, p) / p + 0.2
    X *= D[None,:]
    Y = rng.standard_normal(n) * 2
    beta = np.zeros(p)
    if alt:
        subs = rng.choice(p, s, replace=False)
        beta[subs] = rng.standard_normal(s) 
    mu = X @ beta
    Y += mu

    if family == 'gaussian':
        fam = sm.families.Gaussian()
    elif family == 'probit':
        fam = sm.families.Binomial(link=sm.families.links.Probit())
        Y = (mu + rng.standard_normal(n)) > 0
    else:
        raise ValueError('only testing "gaussian" and "probit"')
    
    if penalty_facs:
        penalty_factor = np.ones(X.shape[1])
        penalty_factor[:10] = 0.5
    else:
        penalty_factor = None

    GN = GLMNet(response_id='response',
                family=fam,
                fit_intercept=fit_intercept,
                standardize=standardize,
                penalty_factor=penalty_factor)

    Df = pd.DataFrame({'response':Y})

    GN.fit(X, Df)
    m = int(prop*n)

    df = lasso_inference(GN, 
                         GN.lambda_values_[min(10, GN.lambda_values_.shape[0]-1)],
                         (X[:m], Df.iloc[:m], None),
                         (X, Df, None))
    if fit_intercept:
        active_set = np.array(df.index[1:]).astype(int)
    else:
        active_set = np.array(df.index).astype(int)
    X_sel = X[:,active_set]

    if fit_intercept:
        X_sel = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
    targets = np.linalg.pinv(X_sel) @ mu

    df['target'] = targets
    if alt:
        if family == 'probit' and not set(subs).issubset(active_set):
            df['target'] *= np.nan
            
    return df


def main_null(fit_intercept=True,
              standardize=True,
              n=500,
              p=75,
              ntrial=500,
              rng=None,
              family='gaussian'):

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
                           family=family)
        dfs.append(df)
        all_df = pd.concat(dfs)
        ncover += ((all_df['lower'] < 0) & (all_df['upper'] > 0)).sum()
        nsel += all_df.shape[0]

        print('cover:', ncover / nsel, 'typeI:', (all_df['pval'] < 0.05).mean())


def main_alt(fit_intercept=True,
              standardize=True,
              n=1000,
              p=75,
              ntrial=500,
              rng=None,
              family='gaussian'):

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
                           family=family,
                           alt=True)
        dfs.append(df)
        all_df = pd.concat(dfs).dropna()
        ncover += ((all_df['lower'] < all_df['target']) & (all_df['upper'] > all_df['target'])).sum()
        nsel += all_df.shape[0]

        print('cover:', ncover / nsel)
    return all_df

        
