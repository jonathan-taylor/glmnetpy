"""
This module contains the core code needed for post selection
after LASSO.

"""

import warnings
from warnings import warn
from copy import copy

import numpy as np
import scipy.sparse
import pandas as pd
from scipy.stats import norm as normal_dbn
import statsmodels.api as sm

import mpmath as mp
mp.dps = 80

from .base import _get_design
from .glmnet import GLMNet

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TruncatedGaussian(object):
    
    estimate: float
    sigma: float
    smoothing_sigma: float
    lower_bound: float
    upper_bound: float
    level: Optional[float] = 0.9
    num_sd: Optional[float] = 10
    num_grid: Optional[float] = 4000
    use_sample: Optional[bool] = False
    seed: Optional[int] = 0

    def _get_family(self,
                    basept=None):
   
        if basept is None:
            basept = self.estimate
            
        self._rng = np.random.default_rng(self.seed)
        
        if not self.use_sample:
            grid = np.linspace(basept - self.num_sd * self.sigma,
                               basept + self.num_sd * self.sigma, self.num_grid)

            weight = (normal_dbn.cdf((self.upper_bound - grid) / self.smoothing_sigma)
                      - normal_dbn.cdf((self.lower_bound - grid) / self.smoothing_sigma))

            weight *= normal_dbn.pdf((grid - basept) / self.sigma)

            return discrete_family(grid, weight)

        else:
            sample = self._rng.standard_normal(self.num_grid) * self.sigma + basept
            weight = (normal_dbn.cdf((self.upper_bound - sample) / self.smoothing_sigma)
                      - normal_dbn.cdf((self.lower_bound - sample) / self.smoothing_sigma))
            return discrete_family(sample, weight)

    def pvalue(self,
               null_value=0,
               alternative='twosided',
               basept=None):

        if basept is None:
            basept = null_value

        if alternative not in ['twosided', 'greater', 'less']:
            raise ValueError("alternative should be one of ['twosided', 'greater', 'less']")
        _family = self._get_family(basept=basept)
        tilt = (null_value - basept) / self.sigma**2
        if alternative in ['less', 'twosided']:
            _cdf = _family.cdf(tilt, x=self.estimate)
            if alternative == 'less':
                pvalue = _cdf
            else:
                pvalue = 2 * min(_cdf, 1 - _cdf)
        else:
            pvalue = _family.sf(tilt, x=self.estimate)
        return pvalue

    def interval(self,
                 basept=None):
                 
        if basept is None:
            basept = self.estimate
            
        _family = self._get_family(basept=basept)

        L, U = _family.equal_tailed_interval(self.estimate,
                                             alpha=1-self.level)
        L *= self.sigma**2; L += basept
        U *= self.sigma**2; U += basept

        return L, U

    def MLE(self,
            basept=None):

        if basept is None:
            basept = self.estimate
            
        _family = self._get_family(basept=basept)

        mle, _, _ = _family.MLE(self.estimate)
        mle *= self.sigma**2; mle += basept
        return mle
    
@dataclass
class AffineConstraint(object):

    linear: np.ndarray
    offset: np.ndarray
    observed: np.ndarray

    def __post_init__(self):

        if not all (self.linear @ self.observed - self.offset <= 0):
            raise ValueError('constraint not satisfied for observed data')
    
    def interval_constraints(self,
                             target,
                             gamma,
                             tol = 1.e-4):
        r"""
        Given an affine constraint $\{z:Az \leq b \leq \}$ (elementwise)
        specified with $A$ as `support_directions` and $b$ as
        `support_offset`, a new direction of interest $\eta$, and
        an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
        with `covariance` matrix $\Sigma$, this
        function returns $\eta^TZ$ as well as an interval
        bounding this value. 

        The interval constructed is such that the endpoints are 
        independent of $\eta^TZ$, hence the $p$-value
        of `Kac Rice`_
        can be used to form an exact pivot.

        Parameters
        ----------

        support_directions : np.float
             Matrix specifying constraint, $A$.

        support_offsets : np.float
             Offset in constraint, $b$.

        covariance : np.float
             Covariance matrix of `observed_data`.

        observed_data : np.float
             Observations.

        direction_of_interest : np.float
             Direction in which we're interested for the
             contrast.

        tol : float
             Relative tolerance parameter for deciding 
             sign of $Az-b$.

        Returns
        -------

        lower_bound : float

        observed : float

        upper_bound : float

        sigma : float

        """

        # shorthand
        A, b, D = (self.linear,
                   self.offset,
                   self.observed)

        N = D - gamma * target

        U = A.dot(N) - b
        V = A @ gamma
        
        # Inequalities are now U + V @ target <= 0
        
        # adding the zero_coords in the denominator ensures that
        # there are no divide-by-zero errors in RHS
        # these coords are never used in upper_bound or lower_bound

        zero_coords = V == 0
        RHS = -U / (V + zero_coords)
        RHS[zero_coords] = np.nan
        pos_coords = V > tol * np.fabs(V).max()
        neg_coords = V < -tol * np.fabs(V).max()

        if np.any(pos_coords):
            upper_bound = RHS[pos_coords].min()
        else:
            upper_bound = np.inf

        if np.any(neg_coords):
            lower_bound = RHS[neg_coords].max()
        else:
            lower_bound = -np.inf

        return lower_bound, upper_bound

def lasso_inference(glmnet_obj,
                    lambda_val,
                    selection_data,
                    full_data,
                    level=.9,
                    dispersion=None):

    fixed_lambda, warm_state = glmnet_obj.get_fixed_lambda(lambda_val)
    X_sel, Y_sel, weight_sel = selection_data
    X_sel, Y_sel, _, _, weight_sel = fixed_lambda.get_data_arrays(X_sel,
                                                                  Y_sel)
    if weight_sel is None:
        weight_sel = np.ones(X_sel.shape[0])

    fixed_lambda.fit(X_sel,
                     Y_sel,
                     sample_weight=weight_sel,
                     warm_state=warm_state)

    FL = fixed_lambda # shorthand

    if (not np.all(FL.upper_limits >= FL.control.big) or
        not np.all(FL.lower_limits <= -FL.control.big)):
        raise NotImplementedError('upper/lower limits coming soon')

    active_set = np.nonzero(FL.coef_ != 0)[0]

    if active_set.shape[0] != 0:
        inactive_set = np.nonzero(FL.coef_ == 0)[0]

        # fit unpenalized model on selection data

        unreg_sel_GLM = glmnet_obj.get_GLM()

        unreg_sel_GLM.summarize = True
        unreg_sel_GLM.fit(X_sel[:,active_set],
                          Y_sel,
                          sample_weight=weight_sel,
                          dispersion=dispersion)

        # # quadratic approximation up to scaling and a factor of weight_sel.sum()

        D_sel = _get_design(X_sel[:,active_set],
                            weight_sel,
                            standardize=glmnet_obj.standardize,
                            intercept=glmnet_obj.fit_intercept)
        P_noisy = D_sel.quadratic_form(unreg_sel_GLM._information,
                                       transformed=True)

        # fit unpenalized model on full data

        X_full, Y_full, weight_full = full_data
        if weight_full is None:
            weight_full = np.ones(X_full.shape[0])

        unreg_GLM = glmnet_obj.get_GLM()
        unreg_GLM.summarize = True
        unreg_GLM.fit(X_full[:,active_set],
                      Y_full,
                      sample_weight=weight_full,
                      dispersion=dispersion)

        # quadratic approximation

        D = _get_design(X_full[:,active_set],
                        weight_full,
                        standardize=glmnet_obj.standardize,
                        intercept=glmnet_obj.fit_intercept)

        if np.asarray(FL.upper_limits).shape == ():
            upper_limits = np.ones(D_sel.shape[1] - 1) * FL.upper_limits
        else:
            upper_limits = FL.upper_limits[active_set]
            
        if np.asarray(FL.lower_limits).shape == ():
            lower_limits = np.ones(D_sel.shape[1] - 1) * FL.lower_limits
        else:
            lower_limits = FL.lower_limits[active_set]
            
        upper_limits = D_sel.scaling_ * upper_limits
        lower_limits = D_sel.scaling_ * lower_limits

        P_full = D.quadratic_form(unreg_GLM._information,
                                  transformed=True)
        if not FL.fit_intercept:
            if FL.penalty_factor is not None:
                penfac = FL.penalty_factor[active_set]
            else:
                penfac = np.ones_like(active_set)
            P_full = P_full[1:,1:]
            P_noisy = P_noisy[1:,1:]
        else:
            penfac = np.ones(active_set.shape[0])

        Q_full = np.linalg.inv(P_full) 
        Q_noisy = np.linalg.inv(P_noisy)

        signs = np.sign(FL.coef_[active_set])

        noisy_mle = D.raw_to_scaled(unreg_sel_GLM.state_)
        
        if FL.fit_intercept:
            penfac = np.hstack([0, penfac])
            signs = np.hstack([0, signs])
            stacked = np.hstack([FL.state_.intercept,
                                 FL.state_.coef[active_set]])
            noisy_mle = noisy_mle._stack
        else:
            stacked = FL.state_.coef[active_set]
            noisy_mle = noisy_mle.coef

        # delta = lambda_val * penfac * signs

        # # remember loss of glmnet is normalized by sum of weights
        # # when taking newton step adjust by weight_sel.sum()

        # delta = Q_sel @ delta * weight_sel.sum() 
        # noisy_mle = stacked + delta # unitless scale

        penalized = penfac > 0
        n_penalized = penalized.sum()
        n_coef = penalized.shape[0]
        row_idx = np.arange(n_penalized)
        col_idx = np.nonzero(penalized)[0]
        data = -signs[penalized]
        sel_P = scipy.sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_penalized, n_coef))

        if FL.fit_intercept:
            sel_U = sel_L = scipy.sparse.dia_array((np.ones((1, n_coef)), [1]), shape=(n_coef-1, n_coef))
        else:
            sel_U = sel_L = scipy.sparse.eye(n_coef)

        ## the GLM's coef and intercept are on the original scale
        ## we transform them here to the (typically) unitless "standardized" scale

        full_mle = D.raw_to_scaled(unreg_GLM.state_)

        if FL.fit_intercept:
            full_mle = full_mle._stack
        else:
            full_mle = full_mle.coef

        ## iterate over coordinates
        Ls = np.zeros_like(stacked)
        Us = np.zeros_like(stacked)
        mles = np.zeros_like(stacked)
        pvals = np.zeros_like(stacked)
        TGs = []
        
        transform_to_raw = (D.unscaler_ @
                            np.identity(D.shape[1]))
        if not FL.fit_intercept:
            transform_to_raw = transform_to_raw[1:,1:]

        linear = scipy.sparse.vstack([sel_P, sel_U, -sel_L])
        offset = np.hstack([np.zeros(sel_P.shape[0]), upper_limits, -lower_limits]) 
        active_con = AffineConstraint(linear=linear,
                                      offset=offset,
                                      observed=stacked)
        inactive = True
        if inactive:
            pf = FL.regularizer_.penalty_factor

            logl_score = FL.state_.logl_score(FL._family,
                                              Y_sel,
                                              weight_sel / weight_sel.sum())

            # X_{-E}'(Y-X\hat{\beta}_E) -- \hat{\beta}_E is the LASSO soln, not GLM !
            # this is (2nd order Taylor series sense) equivalent to
            # X_{-E}'(Y-X\bar{\beta}_E) + X_{-E}'WX_E(X_E'WX_E)^{-1}\lambda_E s_E
            # with \bar{\beta}_E the GLM soln

            score_ = (FL.design_.T @ logl_score)[1:]
            score_ = score_[inactive_set]
            # we now know that `score_` is bounded by \pm lambda_val

            I = scipy.sparse.eye(score_.shape[0])
            L = scipy.sparse.vstack([I, -I])
            O = (np.ones(L.shape[0]) * lambda_val *
                 np.hstack([pf[inactive_set],
                            pf[inactive_set]]))

            fudge_factor = 0.02 # allow 2% relative error on inactive gradient
            inactive_con = AffineConstraint(linear=L,
                                            offset=O * (1 + fudge_factor),
                                            observed=score_)

        for i in range(transform_to_raw.shape[0]):
            ## call selection_interval and return
            TG = _split_interval(active_con=active_con,
                                 Q_noisy=Q_noisy * unreg_GLM.dispersion_,
                                 Q_full=Q_full * unreg_GLM.dispersion_,
                                 noisy_observation=noisy_mle, # these must be same scale / shift as glmnet
                                 observation=full_mle, # these must be same scale / shift as glmnet
                                 direction_of_interest=transform_to_raw[i], # this matrix describes the map from glmnet("transform") coords to raw("original") scale
                                 level=level)

            (L, U), mle, pval = (TG.interval(), TG.MLE(), TG.pvalue())

            Ls[i] = L
            Us[i] = U
            mles[i] = mle
            pvals[i] = pval
            TGs.append(TG)
            
        idx = active_set.tolist()
        if FL.fit_intercept:
            idx = ['intercept'] + idx

        TG_df = pd.concat([pd.Series(asdict(tg)) for tg in TGs], axis=1).T
        TG_df.index = idx
        df = pd.concat([pd.DataFrame({'mle': mles, 'pval': pvals, 'lower': Ls, 'upper': Us}, index=idx), TG_df], axis=1)
        df['TG'] = TGs
        return df
    else:
        return None

def _split_interval(active_con,
                    Q_noisy,
                    Q_full,
                    noisy_observation,
                    observation,
                    direction_of_interest,
                    tol = 1.e-4,
                    level = 0.90,
                    dispersion=1):
    """
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    `noisy_observation` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns a confidence interval
    for $\eta^T\mu$.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    noisy_observation : np.float
         Observations.

    observation : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    Returns
    -------

    confidence_interval : (float, float)

    """

    noisy_var = direction_of_interest.T @ Q_noisy @ direction_of_interest
    full_var = direction_of_interest.T @ Q_full @ direction_of_interest
    noisy_estimate = (direction_of_interest * noisy_observation).sum()
    estimate = (direction_of_interest * observation).sum()

    slice_dir = Q_full @ direction_of_interest / full_var
    (lower_bound,
     upper_bound) = active_con.interval_constraints(noisy_estimate,
                                                    slice_dir,
                                                    tol=tol)

    sigma = np.sqrt(full_var)
    smoothing_sigma = np.sqrt(max(noisy_var - full_var, 0))

    return TruncatedGaussian(estimate=estimate,
                             sigma=sigma,
                             smoothing_sigma=smoothing_sigma,
                             lower_bound=lower_bound,
                             upper_bound=upper_bound,
                             level=level)
    
def _norm_interval(lower, upper):
    r"""
    A multiprecision evaluation of

    .. math::

        \Phi(U) - \Phi(L)

    Parameters
    ----------

    lower : float
        The lower limit $L$

    upper : float
        The upper limit $U$

    """
    #cdf = normal_dbn.cdf
    cdf = mp.ncdf
    if lower > 0:
        if lower < 4:
            return cdf(-lower) - cdf(-upper)
        else:
            return np.exp(-np.fabs(lower*(upper-lower)))
    elif upper < 0:
        if upper < -4:
            return np.exp(-np.fabs(upper*(upper-lower)))
        else:
            return cdf(-lower) - cdf(-upper)
    else:
        return cdf(upper) - cdf(lower)

def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    
    # determine the necessary number of iterations
    try:
        max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )
    except OverflowError:
        warnings.warn('root finding failed, returning np.nan')
        return np.nan
        

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        try:
            c, fc  = (a+b)/2, f((a+b)/2)
            if fc > y: a = c
            elif fc < y: b = c
        except OverflowError:
            warnings.warn('root finding failed, returning np.nan')
            return np.nan

    return c
        
class discrete_family(object):

    def __init__(self, sufficient_stat, weights, theta=0.):
        r"""
        A  discrete 1-dimensional
        exponential family with reference measure $\sum_j w_j \delta_{X_j}$
        and sufficient statistic `sufficient_stat`. For any $\theta$, the distribution
        is

        .. math::
        
            P_{\theta} = \sum_{j} e^{\theta X_j - \Lambda(\theta)} w_j \delta_{X_j}

        where

        .. math::

            \Lambda(\theta) = \log \left(\sum_j w_j e^{\theta X_j} \right).

        Parameters
        ----------

        sufficient_stat : `np.float((n))`

        weights : `np.float(n)`

        Notes
        -----

        The weights are normalized to sum to 1.
        """
        xw = np.array(sorted(zip(sufficient_stat, weights)), float)
        xw[:,1] = np.maximum(xw[:,1], 1e-40)
        self._x = xw[:,0]
        self._w = xw[:,1]
        self._lw = np.log(xw[:,1])
        
        self._w /= self._w.sum() # make sure they are a pmf
        self.n = len(xw)
        self._theta = np.nan
        self.theta = theta

    @property
    def theta(self):
        """
        The natural parameter of the family.
        """
        return self._theta

    @theta.setter
    def theta(self, _theta):
        if _theta != self._theta:
            _thetaX = _theta * self.sufficient_stat + self._lw
            _largest = _thetaX.max() - 15 # try to avoid over/under flow, 5 seems arbitrary
            _exp_thetaX = np.exp(_thetaX - _largest)
            _prod = _exp_thetaX
            self._partition = np.sum(_prod)
            self._pdf = _prod / self._partition
            self._partition *= np.exp(_largest)
        self._theta = _theta

    @property
    def partition(self):
        r"""
        Partition function at `self.theta`:

        .. math::

            \sum_j e^{\theta X_j} w_j
        """
        if hasattr(self, "_partition"):
            return self._partition

    @property
    def sufficient_stat(self):
        """
        Sufficient statistics of the exponential family.
        """
        return self._x

    @property
    def weights(self):
        """
        Weights of the exponential family.
        """
        return self._w

    def pdf(self, theta):
        r"""
        Density of $P_{\theta}$ with respect to $P_0$.

        Parameters
        ----------

        theta : float
             Natural parameter.

        Returns
        -------

        pdf : np.float
        
        """
        self.theta = theta # compute partition if necessary
        return self._pdf
 
    def cdf(self, theta, x=None, gamma=1):
        r"""
        The cumulative distribution function of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X < x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate CDF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        cdf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf) - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat < x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

    def sf(self, theta, x=None, gamma=0, return_unnorm=False):
        r"""
        The complementary cumulative distribution function 
        (i.e. survival function) of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X > x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate SF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        sf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf[::-1])[::-1] - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat > x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

    def E(self, theta, func):
        r"""
        Expectation of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        E : np.float

        """
        T = np.asarray(func(self.sufficient_stat))
        pdf_ = self.pdf(theta)

        if T.ndim == 1:
            return (T * pdf_).sum()
        else:
            val = (T * pdf_[:,None]).sum(0)
            return val


    def Var(self, theta, func):
        r"""
        Variance of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        Returns
        -------

        var : np.float

        """

        mu = self.E(theta, func)
        return self.E(theta, lambda x: (func(x)-mu)**2)
        
    def Cov(self, theta, func1, func2):
        r"""
        Covariance of `func1` and `func2` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func1, func2 : callable
             Assumed to be vectorized.

        Returns
        -------

        cov : np.float

        """

        mu1 = self.E(theta, func1)
        mu2 = self.E(theta, func2)
        return self.E(theta, lambda x: (func1(x)-mu1)*(func2(x)-mu2))

    def equal_tailed_interval(self,
                              observed,
                              alpha=0.05,
                              randomize=True,
                              auxVar=None,
                              tol=1e-6):
        """
        Form interval by inverting
        equal-tailed test with $\alpha/2$ in each tail.

        Parameters
        ----------

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        lower, upper : float
             Limits of confidence interval.

        """

        mu = self.E(self.theta, lambda x: x)
        sigma  = np.sqrt(self.Var(self.theta, lambda x: x))
        lb = mu - 20 * sigma
        ub = mu + 20 * sigma
        F = lambda th : self.cdf(th, observed)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = find_root(F, 0.5 * alpha, lb, ub)
        return L, U

    def equal_tailed_test(self,
                          theta0,
                          observed,
                          alpha=0.05):
        r"""
        Equal tailed test of H_0:theta=theta_0

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        Returns
        -------

        decision : np.bool
             Is the null hypothesis $H_0:\theta=\theta_0$ rejected?
   
        Notes
        -----

        We need an auxiliary uniform variable to carry out the randomized test.
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
        or chosen at random. If randomize=False, we get a conservative test.
        """

        pval = self.cdf(theta0, observed, gamma=0.5)
        return min(pval, 1-pval) < alpha

    def MLE(self,
            observed,
            initial=0,
            max_iter=20,
            tol=1.e-4):

        r"""
        Compute the maximum likelihood estimator
        based on observed sufficient statistic `observed`.

        Parameters
        ----------

        observed : float
             Observed value of sufficient statistic

        initial : float
             Starting point for Newton-Raphson

        max_iter : int (optional)
             Maximum number of Newton-Raphson iterations

        tol : float (optional)
             Tolerance parameter for stopping, based
             on relative change in parameter estimate.
             Iteration stops when the change is smaller
             than `tol * max(1, np.fabs(cur_estimate))`.

        Returns
        -------

        theta_hat : float
             Maximum likelihood estimator.
   
        std_err : float
             Estimated variance of `theta_hat` based
             on inverse of variance of sufficient
             statistic at `theta_hat`, i.e. the
             observed Fisher information.

        """

        cur_est = initial

        def first_two_moments(x):
            return np.array([x, x**2]).T
        
        for i in range(max_iter):
            cur_moments = self.E(cur_est, first_two_moments) # gradient and
                                                             # Hessian of CGF
                                                             # (almost)
            grad, hessian = (cur_moments[0] - observed, 
                             cur_moments[1] - cur_moments[0]**2)
            next_est = cur_est - grad / hessian # newton step

            if np.fabs(next_est - cur_est) < tol * max(1, np.fabs(cur_est)):
                break
            cur_est = next_est

            if i == max_iter - 1:
                warnings.warn('Newton-Raphson failed to converge after %d iterations' % max_iter)

        cur_moments = self.E(cur_est, first_two_moments) # gradient and
                                                         # Hessian of CGF
                                                         # (almost)
        grad, hessian = (cur_moments[0] - observed, 
                         cur_moments[1] - cur_moments[0]**2)

        return cur_est, 1. / hessian, grad

def _truncated_inference(estimate,
                         sigma,
                         smoothing_sigma,
                         lower_bound,
                         upper_bound,
                         basept=None,
                         level=0.9):

    if basept is None:
        basept = estimate
    noisy_sigma = np.sqrt(smoothing_sigma**2 + sigma**2)

    if smoothing_sigma > 1e-6 * sigma:

        grid = np.linspace(basept - 10 * sigma, basept + 10 * sigma, 2001)

        weight = (normal_dbn.cdf((upper_bound - grid) / smoothing_sigma)
                  - normal_dbn.cdf((lower_bound - grid) / smoothing_sigma))

        weight *= normal_dbn.pdf((grid - basept) / sigma)

        sel_distr = discrete_family(grid, weight)

        use_sample = False
        if use_sample:
            rng = np.random.default_rng(0)
            sample = rng.standard_normal(2000) * sigma + basept
            weight_s = (normal_dbn.cdf((upper_bound - sample) / smoothing_sigma)
                      - normal_dbn.cdf((lower_bound - sample) / smoothing_sigma))
            sel_distr =  discrete_family(sample, weight_s)

        L, U = sel_distr.equal_tailed_interval(estimate,
                                               alpha=1-level)
        mle, _, _ = sel_distr.MLE(estimate)
        mle *= sigma**2; mle += estimate
        pval = sel_distr.cdf(-basept / sigma**2, estimate)
        pval = 2 * min(pval, 1-pval)
        L *= sigma**2; L += basept
        U *= sigma**2; U += basept

    else:
        warnings.warn('assuming data for selection is same as for inference -- using hard selection')
        
        lb = estimate - 20 * sigma
        ub = estimate + 20 * sigma

        if estimate < lower_bound or estimate > upper_bound:
            warn('Constraints not satisfied: returning [-np.inf, np.inf]')
            return -np.inf, np.inf, np.nan, np.nan
        
        def F(theta):
            
            Z = (estimate - theta) / sigma
            Z_L = (lower_bound - theta) / sigma
            Z_U = (upper_bound - theta) / sigma
            num = _norm_interval(Z_L, Z)
            den = _norm_interval(Z_L, Z_U)
            if Z_L > 0 and den < 1e-10:
                C = np.fabs(Z_L)
                D = Z-Z_L
                cdf = 1-np.exp(-C*D)
            elif Z_U < 0 and den < 1e-10:
                C = np.fabs(Z_U)
                D = Z_U-Z
                if C*D < 0:
                    raise ValueError
                cdf = np.exp(-C*D)
            else:
                cdf = num / den
            return cdf
            
        
        pval = F(0)
        pval = 2 * min(pval, 1-pval)

        lb = lower_bound - 20 * sigma
        ub = upper_bound + 20 * sigma

        alpha = 0.5 * (1 - level)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        if np.isnan(L):
            L = -np.inf
        U = find_root(F, 0.5 * alpha, lb, ub)
        if np.isnan(U):
            U = np.inf

        mle = np.nan
        
    return L, U, mle, pval, sel_distr

def score_inference(score,
                    cov_score,
                    lamval,
                    prop=0.8,
                    chol_cov=None,
                    perturbation=None,
                    rng=None):

    # perturbation should be N(0, cov_score) roughly -- shouldn't have a multiplier for the proportion!

    # this is the X for the LASSO problem
    # X.T @ X = S = cov_score

    if chol_cov is None:
        chol_cov = X = np.linalg.cholesky(cov_score).T 

    if rng is None:
        rng = np.random.default_rng()

    # shorthand
    Z = score # X.T @ Y in OLS case
    X = chol_cov 
    p = X.shape[1]
    
    if perturbation is None:
        perturbation = X.T @ rng.standard_normal(p) # X is square here...

    # this is the Y of the LASSO problem

    Y = scipy.linalg.solve_triangular(chol_cov.T, Z, lower=True)
    Df = pd.DataFrame({'response':Y})

    GN = GLMNet(response_id='response',
                fit_intercept=False,
                standardize=False)
    GN.fit(X, Df)
    
    Z_sel = Z + np.sqrt((1 - prop) / prop) * perturbation
    Y_sel = scipy.linalg.solve_triangular(chol_cov.T, Z_sel, lower=True)
    Df_sel = pd.DataFrame({'response':Y_sel * np.sqrt(prop)})
    X_sel = np.sqrt(prop) * X

    return lasso_inference(GN, 
                           prop * lamval / p,
                           (X_sel, Df_sel, None),
                           (X, Df, None),
                           dispersion=1)

def resampler_inference(sample,
                        lamval=None,
                        lam_frac=1,
                        prop=0.8,
                        random_idx=None,
                        rng=None,
                        estimate=None,
                        standardize=True):

    if estimate is None:
        estimate = sample.mean(0)

    centered = sample - estimate[None,:]
    B, p = centered.shape

    if random_idx is None:
        if rng is None:
            rng = np.random.default_rng()
        random_idx = rng.choice(B, 1)
        
    prec_score = centered.T @ centered / B
    cov_score = np.linalg.inv(prec_score)

    centered_scores = centered @ cov_score
    scaling = centered_scores.std(0)
    score = cov_score @ estimate

    if standardize:
        centered_scores /= scaling[None,:]
        cov_score /= np.multiply.outer(scaling, scaling)
        score /= scaling
        
    # pick a lam
    if lamval is None:
        max_scores = np.fabs(centered_scores).max(1)
        lamval = lam_frac * np.median(max_scores)

    # pick a pseudo-Gaussian perturbation
    perturbation = centered_scores[random_idx].reshape((p,))
    
    df = score_inference(score=score,
                         cov_score=cov_score,
                         lamval=lamval,
                         prop=prop,
                         perturbation=perturbation)

    if df is not None:
        if standardize:
            for col in ['mle', 'upper', 'lower']:
                df[col].values[:] = df[col] / scaling[df.index]

            for j in df.index:
                df.loc[j,'TG'] = None
                # XX try to correct this!!
                # tg = df.loc[j, 'TG']
                # tg_params = asdict(tg)
                # tg_params.update(estimate=tg.estimate * scaling[j],
                #                  sigma=tg.sigma * scaling[j],
                #                  smoothing_sigma=tg.smoothing_sigma * scaling[j],
                #                  lower_bound=tg.lower_bound * scaling[j],
                #                  upper_bound=tg.upper_bound * scaling[j])
                # df.loc[j,'TG'] = TruncatedGaussian(**tg_params)

    return df

