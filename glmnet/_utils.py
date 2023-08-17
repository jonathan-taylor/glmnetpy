import numpy as np
import scipy.sparse
import statsmodels.api as sm

def _get_limits(lower_limits,
                upper_limits,
                nvars,
                big):
    lower_limits = np.asarray(lower_limits)
    upper_limits = np.asarray(upper_limits)
    
    if lower_limits.shape in [(), (1,)]:
        lower_limits = -np.inf * np.ones(nvars)

    if upper_limits.shape in [(), (1,)]:
        upper_limits = np.inf * np.ones(nvars)

    lower_limits = lower_limits[:nvars]
    upper_limits = upper_limits[:nvars]

    if lower_limits.shape[0] < nvars:
        raise ValueError('lower_limits should have shape {0}, but has shape {1}'.format((nvars,),
                                                                                        lower_limits.shape))
    if upper_limits.shape[0] < nvars:
        raise ValueError('upper_limits should have shape {0}, but has shape {1}'.format((nvars,),
                                                                                        upper_limits.shape))
    lower_limits[lower_limits == -np.inf] = -big
    upper_limits[upper_limits == np.inf] = big

    return lower_limits, upper_limits

def _get_vp(penalty_factor,
            exclude,
            nvars):

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)
        
    # check and standardize penalty factors (to sum to nvars)
    _isinf_penalty = np.isinf(penalty_factor)
    if np.any(_isinf_penalty):
        exclude.extend(np.nonzero(_isinf_penalty)[0])
        exclude = np.unique(exclude)

    exclude = np.asarray(exclude, int)
        
    if exclude.shape[0] > 0:
        if exclude.max() >= nvars:
            raise ValueError("Some excluded variables out of range")
        penalty_factor[exclude] = 1 # now can change penalty_factor

    vp = np.maximum(0, penalty_factor).reshape((-1,1))
    vp = (vp * nvars / vp.sum())

    return vp, exclude

def _get_eta(X,
             X_scale,
             beta,
             a0):
    if scipy.sparse.issparse(X):
        xm, xs = X_scale
        beta = beta / xs
        eta = X @ beta - np.sum(beta * xm) + a0
    else:
        eta = X @ beta + a0
    return eta

def _jerr_elnetfit(n, maxit, k=None):
    if n == 0:
        fatal = False
        msg = ''
    elif n > 0:
        # fatal error
        fatal = True
        msg =(f"Memory allocation error; contact package maintainer" if n < 7777 else
              "Unknown error")
    else:
        fatal = False
        msg = (f"Convergence for {k}-th lambda value not reached after maxit={maxit}" +
               " iterations; solutions for larger lambdas returned")
    return {'n':n,
            'fatal':fatal,
            'msg':f"Error code {n}:" + msg}

#' Elastic net objective function value
#'
#' Returns the elastic net objective function value.
#'
#' @param y Quantitative response variable.
#' @param mu Model's predictions for \code{y}.
#' @param weights Observation weights.
#' @param family A description of the error distribution and link function to be
#' used in the model. This is the result of a call to a family function.
#' @param lambda A single value for the \code{lambda} hyperparameter.
#' @param alpha The elasticnet mixing parameter, with \eqn{0 \le \alpha \le 1}.
#' @param coefficients The model's coefficients (excluding intercept).
#' @param vp Penalty factors for each of the coefficients.
def _obj_function(y,
                  mu,
                  weights,
                  family,
                  lambda_val,
                  alpha,
                  coefficients,
                  vp):
    return (_dev_function(y, mu, weights, family) / 2 +
            lambda_val * _pen_function(coefficients, alpha, vp))


#' Elastic net penalty value
#'
#' Returns the elastic net penalty value without the \code{lambda} factor.
#'
#' The penalty is defined as
#' \deqn{(1-\alpha)/2 \sum vp_j \beta_j^2 + \alpha \sum vp_j |\beta|.}
#' Note the omission of the multiplicative \code{lambda} factor.
#'
#' @param alpha The elasticnet mixing parameter, with \eqn{0 \le \alpha \le 1}.
#' @param coefficients The model's coefficients (excluding intercept).
#' @param vp Penalty factors for each of the coefficients.
def _pen_function(coefficients,
                  alpha = 1.0,
                  vp = 1.0):
    return np.sum(vp * (alpha * np.abs(coefficients) + (1-alpha)/2 * coefficients**2))


#' Elastic net deviance value
#'
#' Returns the elastic net deviance value.
#'
#' @param y Quantitative response variable.
#' @param mu Model's predictions for \code{y}.
#' @param weights Observation weights.
#' @param family A description of the error distribution and link function to be
#' used in the model. This is the result of a call to a family function.
def _dev_function(y,
                  mu,
                  weights,
                  family):
    return np.sum(family.resid_dev(y, mu, weights))

def _get_start(X,
               X_scale,
               y,
               weights,
               family,
               intercept,
               is_offset,
               offset,
               exclude,
               vp,
               alpha): 

    nobs, nvars = X.shape

    # compute mu and null deviance
    # family = binomial() gives us warnings due to non-integer weights
    # to avoid, suppress warnings
    if intercept:
        if is_offset:
            fit = sm.GLM(y,
                         np.ones((y.shape,1)),
                         family,
                         offset=offset,
                         var_weights=weights)
            mu = fit.fitted
        else:
            mu = np.ones(y.shape) * (weights * y).sum() / weights.sum()

    else:
        mu = family.link.inverse(offset)

    nulldev = _dev_function(y, mu, weights, family)

    # if some penalty factors are zero, we have to recompute mu

    vp_zero = sorted(set(exclude).difference(np.nonzero(vp == 0)[0]))
    if vp_zero:
        tempX = X[:,vp_zero]

        if scipy.sparse.issparse(X):
            tempX = X.toarray()

        if intercept:
            tempX = sm.add_constant(tempX)

        tempfit = sm.GLM(y,
                         tempX,
                         family,
                         offset=offset,
                         var_weights=weights)
        mu = tempfit.fittedvalues

    # compute lambda max
    ju = np.ones(nvars)
    ju[exclude] = 0 # we have already included constant variables in exclude

    r = y - mu
    eta = family.link(mu)
    v = family.variance(mu)
    m_e = family.link.inverse_deriv(eta)
    weights = weights / weights.sum()

    rv = r / v * m_e * weights

    if scipy.sparse.issparse(X):
        xm, xs = X_scale
        g = np.abs(X.T @ rv - np.sum(rv) * xm / xs)
    else:
        g = np.abs(X.T @ rv)

    g = g * ju / (vp + (vp <= 0))
    lambda_max = np.max(g) / max(alpha, 1e-3)

    return {'nulldev':nulldev,
            'mu':mu,
            'lambda_max':lambda_max}


