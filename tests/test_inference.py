import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

import scipy.sparse

from glmnet import GLMNet
from glmnet.inference import (lasso_inference,
                              score_inference,
                              resampler_inference,
                              TruncatedGaussian)

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
                         (X, Df, None),
                         proportion=prop)

def sample_orthogonal(rng=None,
                      p=50,
                      s=5,
                      penalty_facs=False,
                      alt=True,
                      prop=0.8):
    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    X = np.identity(p)
    Y = rng.standard_normal(p) 
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])
    mu = beta
    Y += mu
    Df = pd.DataFrame({'response':Y})

    penalty_factor = np.ones(p)
    if penalty_facs:
        penalty_factor[:p//10] = 0.5
        penalty_factor[p//10:2*p//10] = 0
        
    penalty_factor *= p / penalty_factor.sum()

    GN = GLMNet(response_id='response',
                fit_intercept=False,
                standardize=False,
                penalty_factor=penalty_factor)

    lamval = 2
    Y_sel = Y + np.sqrt((1 - prop) / prop) * rng.standard_normal(Y.shape)
    active_naive = np.nonzero(np.fabs(Y_sel) > lamval)[0]

    Df_sel = pd.DataFrame({'response':Y_sel * np.sqrt(prop)})
    X_sel = np.sqrt(prop) * X
    GN.fit(X, Df)
    active_naive = np.nonzero(np.fabs(Y_sel) > lamval)[0]

    pivots = []
    test_pvals = []

    if active_naive.shape[0] > 0:
        df = lasso_inference(GN, 
                             prop * lamval / p,
                             (X_sel, Df_sel, None),
                             (X, Df, None),
                             proportion=prop,
                             dispersion=1)
        df['target'] = mu[df.index]

        signs = np.sign(Y_sel[df.index])
        assert np.all(Y_sel[df.index] * signs >= lamval * GN.penalty_factor[df.index])

    if df is not None:
        for j, s in zip(df.index, signs):
            if s == 1:
                upper_bound = np.inf
                if penalty_factor is not None:
                    lower_bound = lamval * penalty_factor[j]
                else:
                    lower_bound = lamval
            else:
                if penalty_factor is not None:
                    upper_bound = -lamval * penalty_factor[j]
                else:
                    upper_bound = -lamval
                lower_bound = -np.inf

            tg = df.loc[j,'TG']

            TG = TruncatedGaussian(estimate=Y[j],
                                   sigma=1,
                                   smoothing_sigma=np.sqrt((1 - prop) / prop),
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound,
                                   level=0.90)
            pval = TG.pvalue()
            pivots.append(TG.pvalue(null_value=df.loc[j,'target']))
            test_pvals.append(TG.pvalue(null_value=0))
            assert np.allclose(pval, df.loc[j]['pval'])
        df['pivot_byhand'] = pivots
        df['pval_byhand'] = test_pvals
        assert np.allclose(df['pval'], df['pval_byhand'])

    if df is not None:
        df['pivot'] = [df.loc[j,'TG'].pvalue(df.loc[j, 'target']) for j in df.index]
        assert np.allclose(df['pivot'], df['pivot_byhand'])        

    return df

def resample_orthogonal(rng=None,
                        p=50,
                        s=5,
                        alt=True,
                        prop=0.8,
                        standardize=True,
                        B=2000):

    # when standardize is True, coverage will be fine but pivot won't be
    # this is because the TruncatedGaussian instance is not corrected for scale

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])

    scale = np.linspace(1, 3, p)
    beta_hat = beta + rng.standard_normal(p) * scale
    bootstrap_noise = rng.standard_normal((B, p)) * scale[None,:]
    sample = bootstrap_noise + beta_hat[None,:]

    df = resampler_inference(sample,
                             prop=prop,
                             standardize=standardize)
    if df is not None:
        df['target'] = beta[df.index]
        if not standardize:
            df['pivot'] = [df.loc[j,'TG'].pvalue(df.loc[j,'target']) for j in df.index]
        else:
            # scaling transformation not taken into account so cannot reuse TruncatedGaussian object
            df['pivot'] = np.nan * df['pval']
        return df

def sample_AR1(rho=0.6,               
               rng=None,
               p=100,
               s=5,
               alt=True,
               prop=0.8,
               dispersion=2):

    D = np.fabs(np.subtract.outer(np.arange(p), np.arange(p)))
    dispersion = 2
    S = (rho**D) * dispersion

    return sample_cov(S,
                      rng=rng,
                      p=p,
                      s=s,
                      alt=alt,
                      prop=prop)

def resample_AR1(rng=None,
                 p=50,
                 rho=0.6,
                 s=5,
                 alt=True,
                 prop=0.8,
                 standardize=True,
                 B=2000):

    # when standardize is True, coverage will be fine but pivot won't be
    # this is because the TruncatedGaussian instance is not corrected for scale

    D = np.fabs(np.subtract.outer(np.arange(p), np.arange(p)))
    dispersion = 2
    S_init = (rho**D) * dispersion
    S = np.linalg.inv(S_init)
    
    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])

    S_sqrt = np.linalg.cholesky(S)
    beta_hat = beta + S_sqrt @ rng.standard_normal(p) 
    bootstrap_noise = rng.standard_normal((B, p)) @ S_sqrt.T
    sample = bootstrap_noise + beta_hat[None,:]

    S_i = np.linalg.inv(S)
    mu = S_i @ beta

    df = resampler_inference(sample,
                             prop=prop,
                             standardize=standardize)
    if df is not None:
        prec_E = S_i[df.index][:,df.index]
        df['target'] = np.linalg.inv(prec_E) @ mu[df.index]
        if not standardize:
            df['pivot'] = [df.loc[j,'TG'].pvalue(df.loc[j,'target']) for j in df.index]
        else:
            # scaling transformation not taken into account so cannot reuse TruncatedGaussian object
            df['pivot'] = np.nan * df['pval']
        return df

def sample_cov(S,
               rng=None,
               p=100,
               s=5,
               alt=True,
               prop=0.8,
               lamval=3):

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    S_sqrt = X = np.linalg.cholesky(S).T  # X.T @ X = S

    noise = X.T @ rng.standard_normal(p) 
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])
    mu = S @ beta

    Z = mu + noise

    df = score_inference(score=Z,
                         cov_score=S,
                         lamval=lamval,
                         prop=prop,
                         chol_cov=S_sqrt)
    if df is not None:
        active = list(df.index)
        df['target'] = np.linalg.inv(S[active][:,active]) @ mu[active]

    if df is not None:
        df['pivot'] = [df.loc[j,'TG'].pvalue(df.loc[j, 'target']) for j in df.index]
        
    return df

def sample_randomX(n,
                   p,
                   fit_intercept,
                   standardize,
                   rng=None,
                   alt=False,
                   s=10,
                   snr=1,
                   prop=0.8,
                   penalty_facs=False,
                   cv=True,
                   upper_limits=np.inf):

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    X = rng.standard_normal((n, p))
    D = np.ones(p) # np.linspace(1, p, p) / p + 1
    X *= D[None,:]
    sd = np.fabs(1+np.random.standard_normal())
    Y = rng.standard_normal(n) * sd
    if alt:
        beta[subs] = snr * (rng.uniform(3, 5) * rng.choice([-1,1], size=s, replace=True) / np.sqrt(n))

    mu = X @ beta
    Y += mu

    if penalty_facs:
        penalty_factor = np.ones(p)
        penalty_factor[:10] = 0.5
    else:
        penalty_factor = None

    GN = GLMNet(response_id='response',
                fit_intercept=fit_intercept,
                standardize=standardize,
                penalty_factor=penalty_factor,
                upper_limits=upper_limits)

    m = int(prop*n)

    Df = pd.DataFrame({'response':Y})
    GN.fit(X[:m], Df.iloc[:m]) 

    if cv:
        GN.cross_validation_path(X[:m], Df.iloc[:m])
        lamval = GN.index_best_['Mean Squared Error']
    else:
        eps = rng.standard_normal((X.shape[0], 1000)) * sd
        noise_score = X.T @ eps
        lamval = 1.2 * np.median(np.fabs(noise_score).max(0)) / X.shape[0]
        
    df = lasso_inference(GN, 
                         lamval,
                         (X[:m], Df.iloc[:m], None),
                         (X, Df, None),
                         proportion=prop)

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
            
    if df is not None:
        df['pivot'] = [df.loc[j,'TG'].pvalue(df.loc[j, 'target']) for j in df.index]
        
    return df

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('penalty_facs', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
#@pytest.mark.parametrize('upper_limits', [np.inf, 0.1])
@pytest.mark.parametrize('p', [103])
@pytest.mark.parametrize('n', [500])
def test_randomX(n,
                 p,
                 standardize,
                 fit_intercept,
                 penalty_facs,
                 upper_limits=np.inf):

    for _ in range(5):
        df = None
        while df is None: # make sure it is run
            kwargs = dict(n=n,
                          p=p,
                          standardize=standardize,
                          fit_intercept=fit_intercept,
                          penalty_facs=penalty_facs,
                          cv=False,
                          upper_limits=upper_limits)

            df = main(sample_randomX,
                      kwargs,
                      ntrial=1)

def test_resampler(p=50,
                   standardize=True):

    for _ in range(5):
        df = None
        while df is None: # make sure it is run
            df = main(resample_orthogonal,
                      {'p':p},
                      ntrial=1)


def test_orthogonal(p=100):

    for _ in range(5):
        df = None
        while df is None: # make sure it is run
            df = main(sample_orthogonal,
                      {'p':p},
                      ntrial=1)


def test_AR1(p=100,
             rho=0.6):

    for _ in range(5):
        df = None
        while df is None: # make sure it is run
            df = sample_AR1(p=p,
                            rho=rho)

            

def main(sampler,
         kwargs,
         ntrial=500):

    ncover = 0
    nsel = 0

    dfs = []

    rng = np.random.default_rng(0)
    kwargs.update(rng=rng)
    
    for i in range(ntrial):
        df = sampler(**kwargs)
        if df is not None:
            dfs.append(df)
        if len(dfs) > 0:
            all_df = pd.concat(dfs)
            ncover += ((all_df['lower'] < all_df['target']) & (all_df['upper'] > all_df['target'])).sum()
            nsel += all_df.shape[0]

            print('cover:', ncover / nsel,
                  'power:', (all_df['pval'] < 0.05).mean(),
                  'pivot<0.05:', (all_df['pivot'] < 0.05).mean(),
                  'std(pivot):', all_df['pivot'].std())

    if len(dfs) > 0:
        return all_df
        
def test_truncated_inference(B=1000,
                             smoothing_sigma=np.sqrt(1/3),
                             sigma=1,
                             upper_bound=None,
                             lower_bound=None,
                             alt=False,
                             level=0.9):

    pvals = []
    cover = []

    rng = np.random.default_rng(0)
    
    for _ in range(B):
        mu = rng.standard_normal() + 3
        if not alt:
            mu *= 0
        if lower_bound is None or upper_bound is None:
            lower_bound = rng.uniform(0, 2)
            upper_bound = lower_bound + rng.uniform(3, 5)
        while True:
            Z = rng.standard_normal() * sigma
            Z += mu
            Z_noisy = Z + rng.standard_normal() * smoothing_sigma
            if (Z_noisy > lower_bound) and (Z_noisy < upper_bound):
                TG = TruncatedGaussian(estimate=Z,
                                       sigma=sigma,
                                       smoothing_sigma=smoothing_sigma,
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       level=level)
                
                (L, U), mle, pval = (TG.interval(), TG.MLE(), TG.pvalue())

                cover.append((L<mu) * (U>mu))
                pvals.append(pval)
                print(np.mean(cover), np.mean(np.array(pvals) < 0.05))
                break
            
