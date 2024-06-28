import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from glmnet import GLMNet
from glmnet.inference import (lasso_inference,
                              _truncated_inference)

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
                         2 / n, # GN.lambda_values_[min(10, GN.lambda_values_.shape[0]-1)],
                         (X[:m], Df.iloc[:m], None),
                         (X, Df, None))

def run_inference(n,
                  p,
                  fit_intercept,
                  standardize,
                  rng=None,
                  alt=False,
                  s=3,
                  prop=0.75,
                  penalty_facs=True,
                  family='gaussian',
                  orthogonal=False,
                  cv=False):

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    if not orthogonal:
        X = rng.standard_normal((n, p))
        D = np.linspace(1, p, p) / p + 0.2
        X *= D[None,:]
        Y = rng.standard_normal(n) * np.fabs(1+np.random.standard_normal())
        if alt:
            beta[subs] = rng.uniform(3, 5) * rng.choice([-1,1], size=s, replace=True) / np.sqrt(n)

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
        penalty_factor = np.ones(p)
        penalty_factor[:10] = 0.5
    else:
        penalty_factor = None

    if not orthogonal:
        
        GN = GLMNet(response_id='response',
                    fit_intercept=fit_intercept,
                    standardize=standardize,
                    family=fam,
                    penalty_factor=penalty_factor)

        m = int(prop*n)
        
        Df = pd.DataFrame({'response':Y})
        GN.fit(X[:m], Df.iloc[:m]) 
        if cv:
            GN.cross_validation_path(X[:m], Df.iloc[:m])
            lamval = GN.index_best_['Mean Squared Error']
        else:
            eps = rng.standard_normal((X.shape[0], 1000))
            lamval = 1.2 * np.fabs(X.T @ eps).max() / X.shape[0]
        df = lasso_inference(GN, 
                             lamval,
                             (X[:m], Df.iloc[:m], None),
                             (X, Df, None))

        if df is not None:
            if fit_intercept:
                active_set = np.array(df.index[1:]).astype(int)
            else:
                active_set = np.array(df.index).astype(int)
            X_sel = X[:,active_set]

            if fit_intercept:
                X_sel = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
            targets = np.linalg.pinv(X_sel) @ mu

            df['target'] = targets

    else:

        X = np.identity(p)
        Y = rng.standard_normal(p) 
        Df = pd.DataFrame({'response':Y})
        if alt:
            beta[subs] = rng.standard_normal(s) * 2
        mu = X @ beta
        Y += mu
        
        GN = GLMNet(response_id='response',
                    family=fam,
                    fit_intercept=False,
                    standardize=False,
                    penalty_factor=penalty_factor)


        lamval = 2
        Y_sel = Y + np.sqrt((1 - prop) / prop) * rng.standard_normal(Y.shape)

        active_naive = np.nonzero(np.fabs(Y_sel) > lamval)[0]
        Df_sel = pd.DataFrame({'response':Y_sel * np.sqrt(prop)})
        X_sel = np.sqrt(prop) * X
        GN.fit(X, Df)
        if active_naive.shape[0] > 0:
            df = lasso_inference(GN, 
                                 prop * lamval / p,
                                 (X_sel, Df_sel, None),
                                 (X, Df, None),
                                 dispersion=1)
            df['target'] = mu[df.index]
        else:
            return None
    if alt and df is not None:
        if family == 'probit' and not set(subs).issubset(active_set):
            df['target'] *= np.nan
            
    return df


def main(fit_intercept=True,
         standardize=True,
         n=500,
         p=75,
         ntrial=500,
         rng=None,
         family='gaussian',
         alt=False,
         cv=False,
         orthogonal=False):

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
                           alt=alt,
                           cv=cv,
                           orthogonal=orthogonal)
        if df is not None:
            dfs.append(df)
        if len(dfs) > 0:
            all_df = pd.concat(dfs)
            ncover += ((all_df['lower'] < 0) & (all_df['upper'] > 0)).sum()
            nsel += all_df.shape[0]

            print('cover:', ncover / nsel, 'power:', (all_df['pval'] < 0.05).mean())


        
def test_truncated_inference(B=1000,
                             noisy_sigma=0.5,
                             sigma=1,
                             upper_bound=5,
                             lower_bound=2,
                             alt=False):

    pvals = []
    cover = []

    rng = np.random.default_rng(0)
    
    for _ in range(B):
        mu = rng.standard_normal() + 3
        lower_bound = rng.uniform(0, 2)
        upper_bound = lower_bound + rng.uniform(3, 5)
        while True:
            Z = rng.standard_normal() * sigma
            if alt:
                Z += mu
            Z_noisy = Z + rng.standard_normal() * noisy_sigma
            if (Z_noisy > lower_bound) and (Z_noisy < upper_bound):
                L, U, _, pval, D = _truncated_inference(Z,
                                                        sigma,
                                                        noisy_sigma,
                                                        lower_bound,
                                                        upper_bound,
                                                        basept=Z,
                                                        level=0.90)
                cover.append((L<mu) * (U>mu))
                pvals.append(pval)
                print(np.mean(cover), np.mean(np.array(pvals) < 0.05))
                break
            
    return D
