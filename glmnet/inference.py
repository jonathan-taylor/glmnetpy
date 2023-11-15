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

import mpmath as mp
mp.dps = 80

from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

def fixed_lambda_estimator(glmnet_obj,
                           lambda_val):
    check_is_fitted(glmnet_obj, ["coefs_", "feature_names_in_"])
    estimator = clone(glmnet_obj.reg_glm_est_)
    estimator.lambda_val = lambda_val
    coefs, intercepts = glmnet_obj.interpolate_coefs([lambda_val])
    cls = glmnet_obj.state_.__class__
    state = cls(coefs[0], intercepts[0])
    estimator.regularizer_ = glmnet_obj.reg_glm_est_.regularizer_
    estimator.regularizer_.warm_state = state

    return estimator

def lasso_inference(glmnet_obj,
                    lambda_val,
                    selection_data,
                    full_data,
                    level=.9):

    fixed_lambda = fixed_lambda_estimator(glmnet_obj, lambda_val)
    X_sel, Y_sel, weight_sel = selection_data

    if weight_sel is None:
        weight_sel = np.ones(X_sel.shape[0])
        
    fixed_lambda.fit(X_sel, Y_sel, sample_weight=weight_sel)
    FL = fixed_lambda # shorthand
    
    if (not np.all(FL.upper_limits == np.inf) or
        not np.all(FL.lower_limits == -np.inf)):
        raise NotImplementedError('upper/lower limits coming soon')

    active_set = np.nonzero(FL.coef_ != 0)[0]

    # find selection data covariance

    unreg_sel_LM = glmnet_obj.get_LM()
    unreg_sel_LM.summarize = True
    unreg_sel_LM.fit(X_sel[:,active_set], Y_sel, sample_weight=weight_sel)
    C_sel = unreg_sel_LM.covariance_

    state = FL.state_
    information = FL._family.information(state,
                                         weight_sel)
    
    keep = np.zeros(FL.design_.shape[1], bool)
    if hasattr(FL, 'penalty_factors'):
        penfac = FL.penalty_factors[active_set]
    else:
        penfac = np.ones(active_set.shape[0])
    signs = np.sign(FL.coef_[active_set])

    # we multiply by weight_sel.sum() due to this factor
    # appearing in glmnet objective...
    if FL.fit_intercept:
        penfac = np.hstack([0, penfac])
        signs = np.hstack([0, signs])
        keep[0] = 1
        keep[1 + active_set] = 1
        stacked = np.hstack([state.intercept,
                             state.coef[active_set]])
        delta = np.zeros(keep.sum())
        delta[1:] = weight_sel.sum() * lambda_val * penfac[1:] * signs[1:]
    else:
        keep[active_set] = 1
        stacked = state.coef[active_set]
        delta = weight_sel.sum() * lambda_val * penfac * signs

    delta = C_sel @ delta
    noisy_mle = stacked + delta

    penalized = penfac > 0
    sel_P = -np.diag(signs[penalized]) @ np.eye(C_sel.shape[0])[penalized]

    # fit full model

    X_full, Y_full, weight_full = full_data

    unreg_LM = glmnet_obj.get_LM()
    unreg_LM.summarize = True
    unreg_LM.fit(X_full[:,active_set], Y_full, sample_weight=weight_full)
    C_full = unreg_LM.covariance_

    # selection_proportion = np.clip(np.diag(C_full).sum() / np.diag(C_sel).sum(), 0, 1)
    
    ## TODO: will this handle fit_intercept?
    if FL.fit_intercept:
        full_mle = np.hstack([unreg_LM.intercept_,
                              unreg_LM.coef_])
    else:
        full_mle = unreg_LM.coef_

    ## iterate over coordinates
    Ls = np.zeros_like(noisy_mle)
    Us = np.zeros_like(noisy_mle)
    mles = np.zeros_like(noisy_mle)
    pvals = np.zeros_like(noisy_mle)
    for i in range(len(noisy_mle)):
        e_i = np.zeros_like(noisy_mle)
        e_i[i] = 1.
        ## call selection_interval and return
        L, U, mle, p = selection_interval(
            support_directions=sel_P,
            support_offsets=-signs[penalized] * delta[penalized], 
            covariance_noisy=C_sel,
            covariance_full=C_full,
            noisy_observation=noisy_mle,
            observation=full_mle,
            direction_of_interest=e_i,
            level=level,
        )
        Ls[i] = L
        Us[i] = U
        mles[i] = mle
        pvals[i] = p
    return pd.DataFrame({'mle': mles, 'pval': pvals, 'lower': Ls, 'upper': Us})

class constraints(object):

    r"""
    This class is the core object for affine selection procedures.
    It is meant to describe sets of the form $C$
    where

    .. math::

       C = \left\{z: Az\leq b \right \}

    Its main purpose is to consider slices through $C$
    and the conditional distribution of a Gaussian $N(\mu,\Sigma)$
    restricted to such slices.

    Notes
    -----

    In this parameterization, the parameter `self.mean` corresponds
    to the *reference measure* that is being truncated. It is not the
    mean of the truncated Gaussian.

    >>> positive = constraints(-np.identity(2), np.zeros(2))
    >>> Y = np.array([3, 4.4])
    >>> eta = np.array([1, 1], np.float)
    >>> list(positive.interval(eta, Y))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [4.62...,  10.17...]
    >>> positive.pivot(eta, Y) # doctest: +ELLIPSIS 
    5.187...-07
    >>> list(positive.bounds(eta, Y)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [1.399..., 7.4..., inf, 1.414...]  
    >>> 

    """

    def __init__(self, 
                 linear_part,
                 offset,
                 covariance=None,
                 mean=None,
                 rank=None):
        r"""
        Create a new inequality. 

        Parameters
        ----------

        linear_part : np.float((q,p))
            The linear part, $A$ of the affine constraint
            $\{z:Az \leq b\}$. 

        offset: np.float(q)
            The offset part, $b$ of the affine constraint
            $\{z:Az \leq b\}$. 

        covariance : np.float((p,p))
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float(p)
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        rank : int
            If not None, this should specify
            the rank of the covariance matrix. Defaults
            to self.dim.

        """

        self.linear_part, self.offset = \
            linear_part, np.asarray(offset)
        
        if self.linear_part.ndim == 2:
            self.dim = self.linear_part.shape[1]
        else:
            self.dim = self.linear_part.shape[0]

        if rank is None:
            self.rank = self.dim
        else:
            self.rank = rank

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def __copy__(self):
        r"""
        A copy of the constraints.

        Also copies _sqrt_cov, _sqrt_inv if attributes are present.
        """
        con = constraints(copy(self.linear_part),
                          copy(self.offset),
                          mean=copy(self.mean),
                          covariance=copy(self.covariance),
                          rank=self.rank)
        if hasattr(self, "_sqrt_cov"):
            con._sqrt_cov = self._sqrt_cov.copy()
            con._sqrt_inv = self._sqrt_inv.copy()
            con._rowspace = self._rowspace.copy()
        return con

    def __call__(self, Y, tol=1.e-3):
        r"""
        Check whether Y satisfies the linear
        inequality constraints.
        >>> A = np.array([[1., -1.], [1., -1.]])
        >>> B = np.array([1., 1.])
        >>> con = constraints(A, B)
        >>> Y = np.array([-1., 1.])
        >>> con(Y)
        True
        """
        V1 = self.value(Y)
        return np.all(V1 < tol * np.fabs(V1).max(0))

    def value(self, Y):
        r"""
        Compute $\max(Ay-b)$.
        """
        return (self.linear_part.dot(Y) - self.offset).max()

    def conditional(self,
                    linear_part,
                    value,
                    rank=None):
        """
        Return an equivalent constraint 
        after having conditioned on a linear equality.
        
        Let the inequality constraints be specified by
        `(A,b)` and the equality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        .. math::
           
           AY - E(AY|CY=d)

        Parameters
        ----------

        linear_part : np.float((k,q))
             Linear part of equality constraint, `C` above.

        value : np.float(k)
             Value of equality constraint, `b` above.

        rank : int
            If not None, this should specify
            the rank of `linear_part`. Defaults
            to `min(k,q)`.

        Returns
        -------

        conditional_con : `constraints`
             Affine constraints having applied equality constraint.

        """

        S = self.covariance
        C, d = linear_part, value

        M1 = S.dot(C.T)
        M2 = C.dot(M1)

        if M2.shape:
            M2i = np.linalg.pinv(M2)
            delta_cov = M1.dot(M2i.dot(M1.T))
            delta_mean = M1.dot(M2i.dot(C.dot(self.mean) - d))
        else:
            delta_cov = np.multiply.outer(M1, M1) / M2
            delta_mean = M1 * (C.dot(self.mean) - d) / M2

        if rank is None:
            if len(linear_part.shape) == 2:
                rank = min(linear_part.shape)
            else:
                rank = 1

        return constraints(self.linear_part,
                           self.offset,
                           covariance=self.covariance - delta_cov,
                           mean=self.mean - delta_mean,
                           rank=self.rank - rank)

    def bounds(self, direction_of_interest, Y):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(\mu,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        L : np.float
            Lower truncation bound.

        Z : np.float
            The observed $\eta^TY$

        U : np.float
            Upper truncation bound.

        S : np.float
            Standard deviation of $\eta^TY$.

        
        """
        return interval_constraints(self.linear_part,
                                    self.offset,
                                    self.covariance,
                                    Y,
                                    direction_of_interest)

    def pivot(self, 
              direction_of_interest, 
              Y,
              null_value=None,
              alternative='greater'):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        P : np.float
            $p$-value of corresponding test.

        Notes
        -----

        All of the tests are based on the exact pivot $F$ given
        by the truncated Gaussian distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.

        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")
        L, Z, U, S = self.bounds(direction_of_interest, Y)

        if null_value is None:
            meanZ = (direction_of_interest * self.mean).sum()
        else:
            meanZ = null_value

        P = truncnorm_cdf((Z-meanZ)/S, (L-meanZ)/S, (U-meanZ)/S)

        if alternative == 'greater':
            return 1 - P
        elif alternative == 'less':
            return P
        else:
            return max(2 * min(P, 1-P), 0)

    def interval(self, direction_of_interest, Y,
                 alpha=0.05, UMAU=False):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.
        
        Parameters
        ----------

        direction_of_interest: np.float

            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float

            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alpha : float

            What level of confidence?

        UMAU : bool

            Use the UMAU intervals?

        Returns
        -------

        [L,U] : selection interval

        
        """
        ## THE DOCUMENTATION IS NOT GOOD ! HAS TO BE CHANGED !

        return selection_interval( \
            self.linear_part,
            self.offset,
            self.covariance,
            Y,
            direction_of_interest,
            level=1. - alpha,
            UMAU=UMAU)

    def covariance_factors(self, force=True):
        """
        Factor `self.covariance`,
        finding a possibly non-square square-root.

        Parameters
        ----------

        force : bool
            If True, force a recomputation of
            the covariance. If not, assumes that
            covariance has not changed.

        """
        if not hasattr(self, "_sqrt_cov") or force:

            # original matrix is np.dot(U, (D**2 * U).T)

            U, D = np.linalg.svd(self.covariance)[:2]
            D = np.sqrt(D[:self.rank])
            U = U[:,:self.rank]
        
            self._sqrt_cov = U * D[None,:]
            self._sqrt_inv = (U / D[None,:]).T
            self._rowspace = U

        return self._sqrt_cov, self._sqrt_inv, self._rowspace

    def whiten(self):
        """

        Return a whitened version of constraints in a different
        basis, and a change of basis matrix.

        If `self.covariance` is rank deficient, the change-of
        basis matrix will not be square.

        Returns
        -------

        inverse_map : callable

        forward_map : callable

        white_con : `constraints`

        """
        sqrt_cov, sqrt_inv = self.covariance_factors()[:2]

        new_A = self.linear_part.dot(sqrt_cov)
        den = np.sqrt((new_A**2).sum(1))
        new_b = self.offset - self.linear_part.dot(self.mean)
        new_con = constraints(new_A / den[:,None], new_b / den)

        mu = self.mean.copy()

        def inverse_map(Z): 
            if Z.ndim == 2:
                return sqrt_cov.dot(Z) + mu[:,None]
            else:
                return sqrt_cov.dot(Z) + mu

        forward_map = lambda W: sqrt_inv.dot(W - mu)

        return inverse_map, forward_map, new_con

    def project_rowspace(self, direction):
        """
        Project a vector onto rowspace
        of the covariance.
        """
        rowspace = self.covariance_factors()[-1]
        return rowspace.dot(rowspace.T.dot(direction))

    def solve(self, direction):
        """
        Compute the inverse of the covariance
        times a direction vector.
        """
        sqrt_inv = self.covariance_factors()[1]
        return sqrt_inv.T.dot(sqrt_inv.dot(direction))





def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection. 

    Parameters
    ----------

    cons : [`selection.affine.constraints`_]
         A sequence of constraints.

    Returns
    -------

    intersection : `selection.affine.constraints`_

    Notes
    -----

    Resulting constraint will have mean 0 and covariance $I$.

    """
    ineq, ineq_off = [], []
    for con in cons:
        ineq.append(con.linear_part)
        ineq_off.append(con.offset)

    intersection = constraints(np.vstack(ineq), 
                               np.hstack(ineq_off))
    return intersection

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
                       covariance_noisy,
                       covariance_full,
                       noisy_observation,
                       observation,
                       direction_of_interest,
                       tol = 1.e-4,
                       level = 0.90,
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
     _,
     upper_bound,
     _) = interval_constraints(support_directions, 
                                support_offsets,
                                covariance_full,
                                noisy_observation,
                                direction_of_interest,
                                tol=tol)
    ## lars path lockhart tibs^2 taylor paper
    sigma = np.sqrt(direction_of_interest.T @ covariance_full @ direction_of_interest)
    ## sqrt(alpha) is just sigma_noisy - sigma_full
    smoothing_sigma = np.sqrt(max(direction_of_interest.T @ covariance_noisy @ direction_of_interest - sigma**2, 0))
    estimate = (direction_of_interest * observation).sum()

    if smoothing_sigma > 1e-6 * sigma:

        grid = np.linspace(estimate - 20 * sigma, estimate + 20 * sigma, 801)
        weight = (
            normal_dbn.cdf(
                (upper_bound - grid) / (smoothing_sigma)
            ) - normal_dbn.cdf(
                (lower_bound - grid) / (smoothing_sigma)
            )
        )
        weight *= normal_dbn.pdf(grid / sigma)

        # assert(0==1)
        sel_distr = discrete_family(grid, weight)
        L, U = sel_distr.equal_tailed_interval(estimate,
                                                alpha=1-level)
        mle, _, _ = sel_distr.MLE(estimate)
        mle *= sigma**2
        pval = sel_distr.cdf(0, estimate)
        pval = 2 * min(pval, 1-pval)
        L *= sigma**2
        U *= sigma**2
    else:

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

