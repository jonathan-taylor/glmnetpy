"""
This module contains the core code needed for post selection
after LASSO.

"""

import warnings
from warnings import warn
from copy import copy
import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dbn
import statsmodels.api as sm

import mpmath as mp
mp.dps = 80

from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from .base import _get_design

def fixed_lambda_estimator(glmnet_obj,
                           lambda_val):
    check_is_fitted(glmnet_obj, ["coefs_", "feature_names_in_"])
    estimator = glmnet_obj.regularized_estimator(
                           lambda_val=lambda_val,
                           family=glmnet_obj.family,
                           alpha=glmnet_obj.alpha,
                           penalty_factor=glmnet_obj.penalty_factor,
                           lower_limits=glmnet_obj.lower_limits,
                           upper_limits=glmnet_obj.upper_limits,
                           fit_intercept=glmnet_obj.fit_intercept,
                           standardize=glmnet_obj.standardize,
                           control=glmnet_obj.control,
                           offset_id=glmnet_obj.offset_id,
                           weight_id=glmnet_obj.weight_id,            
                           response_id=glmnet_obj.response_id,
                           exclude=glmnet_obj.exclude
                           )

    coefs, intercepts = glmnet_obj.interpolate_coefs([lambda_val])
    cls = glmnet_obj.state_.__class__
    state = cls(coefs[0], intercepts[0])
    return estimator, state

def lasso_inference(glmnet_obj,
                    lambda_val,
                    selection_data,
                    full_data,
                    level=.9):

    fixed_lambda, warm_state = fixed_lambda_estimator(glmnet_obj, lambda_val)
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

    # fit unpenalized model on selection data

    unreg_sel_GLM = glmnet_obj.get_GLM()

    unreg_sel_GLM.summarize = True
    unreg_sel_GLM.fit(X_sel[:,active_set],
                      Y_sel,
                      sample_weight=weight_sel)

    # quadratic approximation up to scaling and a factor of weight_sel.sum()

    design_sel = _get_design(X_sel[:,active_set],
                             weight_sel,
                             standardize=glmnet_obj.standardize,
                             intercept=glmnet_obj.fit_intercept)
    scaling_sel = design_sel.scaling_
    P_sel = design_sel.quadratic_form(unreg_sel_GLM._information,
                                      transformed=True)
    Q_sel = np.linalg.inv(P_sel)
    if not FL.fit_intercept:
        Q_sel = Q_sel[1:,1:]
        
    # fit unpenalized model on full data

    X_full, Y_full, weight_full = full_data
    if weight_full is None:
        weight_full = np.ones(X_full.shape[0])
        
    unreg_GLM = glmnet_obj.get_GLM()
    unreg_GLM.summarize = True
    unreg_GLM.fit(X_full[:,active_set], Y_full, sample_weight=weight_full)

    # quadratic approximation

    design_full = _get_design(X_full[:,active_set],
                              weight_full,
                              standardize=glmnet_obj.standardize,
                              intercept=glmnet_obj.fit_intercept)
    scaling_full = design_full.scaling_
    P_full = design_full.quadratic_form(unreg_GLM._information,
                                        transformed=True)
    Q_full = np.linalg.inv(P_full)
    if not FL.fit_intercept:
        Q_full = Q_full[1:,1:]

    if hasattr(FL, 'penalty_factors'):
        penfac = FL.penalty_factors[active_set]
    else:
        penfac = np.ones(active_set.shape[0])
    signs = np.sign(FL.coef_[active_set])

    if FL.fit_intercept:
        # correct the scaling
        penfac = np.hstack([0, penfac])
        signs = np.hstack([0, signs])
        stacked = np.hstack([FL.state_.intercept,
                             FL.state_.coef[active_set]])
    else:
        stacked = FL.state_.coef[active_set]

    delta = lambda_val * penfac * signs

    # remember loss of glmnet is normalized by sum of weights
    # when taking newton step adjust by weight_sel.sum()
    
    delta = Q_sel @ delta * weight_sel.sum() 
    noisy_mle = stacked + delta # unitless scale

    penalized = penfac > 0
    sel_P = -np.diag(signs[penalized]) @ np.eye(Q_sel.shape[0])[penalized]
    assert (np.all(sel_P @ noisy_mle < -signs[penalized] * delta[penalized]))

    ## the GLM's coef and intercept are on the original scale
    ## we transform them here to the (typically) unitless "standardized" scale
    if FL.fit_intercept:
        intercept = unreg_GLM.state_.intercept + (unreg_GLM.state_.coef * design_full.centers_).sum()
        coef = unreg_GLM.state_.coef * scaling_full
        full_mle = np.hstack([intercept,
                              coef])
    else:
        full_mle = unreg_GLM.state_.coef * scaling_full

    ## iterate over coordinates
    Ls = np.zeros_like(noisy_mle)
    Us = np.zeros_like(noisy_mle)
    mles = np.zeros_like(noisy_mle)
    pvals = np.zeros_like(noisy_mle)

    if FL.fit_intercept:
        transform_to_original = np.diag(np.hstack([1, 1/scaling_full]))
        transform_to_original[0,1:] = -design_full.centers_/scaling_full
    else:
        transform_to_original = np.diag(1/scaling_full)

    for i in range(transform_to_original.shape[0]):
        ## call selection_interval and return
        L, U, mle, p = selection_interval(
            support_directions=sel_P,
            support_offsets=-signs[penalized] * delta[penalized], 
            Q_noisy=Q_sel,
            Q_full=Q_full,
            noisy_observation=noisy_mle,
            observation=full_mle,
            direction_of_interest=transform_to_original[i],
            level=level,
            dispersion=unreg_GLM.dispersion_
        )
        Ls[i] = L
        Us[i] = U
        mles[i] = mle
        pvals[i] = p
    
    idx = (active_set).tolist()
    if FL.fit_intercept:
        idx = ['intercept'] + idx
    return pd.DataFrame({'mle': mles, 'pval': pvals, 'lower': Ls, 'upper': Us}, index=idx)


def interval_constraints(support_directions, 
                         support_offsets,
                         covariance,
                         observed_data, 
                         direction_of_interest,
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
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = A.dot(X) - b

    if not np.all(U  < tol * np.fabs(U).max()):
        stop
        warn('constraints not satisfied: %s' % repr(U))

    Sw = S.dot(w)
    sigma = np.sqrt((w*Sw).sum())
    alpha = A.dot(Sw) / sigma**2
    V = (w*X).sum() # \eta^TZ

    # adding the zero_coords in the denominator ensures that
    # there are no divide-by-zero errors in RHS
    # these coords are never used in upper_bound or lower_bound

    zero_coords = alpha == 0
    RHS = (-U + V * alpha) / (alpha + zero_coords)
    RHS[zero_coords] = np.nan

    pos_coords = alpha > tol * np.fabs(alpha).max()
    if np.any(pos_coords):
        upper_bound = RHS[pos_coords].min()
    else:
        upper_bound = np.inf
    neg_coords = alpha < -tol * np.fabs(alpha).max()
    if np.any(neg_coords):
        lower_bound = RHS[neg_coords].max()
    else:
        lower_bound = -np.inf

    return lower_bound, V, upper_bound, sigma

def selection_interval(support_directions, 
                       support_offsets,
                       Q_noisy,
                       Q_full,
                       noisy_observation,
                       observation,
                       direction_of_interest,
                       tol = 1.e-4,
                       level = 0.90,
                       dispersion=1,
                       UMAU=True):
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

    (lower_bound,
     V,
     upper_bound,
     _) = interval_constraints(support_directions, 
                               support_offsets,
                               Q_noisy,
                               noisy_observation,
                               direction_of_interest,
                               tol=tol)

    ## lars path lockhart tibs^2 taylor paper
    sigma = np.sqrt(direction_of_interest.T @ Q_full @ direction_of_interest * dispersion)
    ## sqrt(alpha) is just sigma_noisy - sigma_full
    noisy_sigma = np.sqrt(direction_of_interest.T @ Q_noisy @ direction_of_interest * dispersion)
    smoothing_sigma = np.sqrt(max(noisy_sigma**2 - sigma**2, 0))
    
    estimate = (direction_of_interest * observation).sum()
    noisy_estimate = (direction_of_interest * noisy_observation).sum()

    if smoothing_sigma > 1e-6 * sigma:

        grid = np.linspace(estimate - 10 * sigma, estimate + 10 * sigma, 2001)
        weight = (
            normal_dbn.cdf(
                (upper_bound - grid) / (smoothing_sigma)
            ) - normal_dbn.cdf(
                (lower_bound - grid) / (smoothing_sigma)
            )
        )
        weight *= normal_dbn.pdf((grid - estimate) / sigma)

        sel_distr = discrete_family(grid, weight)
        L, U = sel_distr.equal_tailed_interval(estimate,
                                               alpha=1-level)
        mle, _, _ = sel_distr.MLE(estimate)
        mle *= sigma**2; mle += estimate
        pval = sel_distr.cdf(-estimate / sigma**2, estimate)
        pval = 2 * min(pval, 1-pval)
        L *= sigma**2; L += estimate
        U *= sigma**2; U += estimate
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
            num = norm_interval(Z_L, Z)
            den = norm_interval(Z_L, Z_U)
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

    return L, U, mle, pval

def norm_interval(lower, upper):
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
    if lower > 0 and upper > 0:
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
            _largest = _thetaX.max() - 5 # try to avoid over/under flow, 5 seems arbitrary
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

