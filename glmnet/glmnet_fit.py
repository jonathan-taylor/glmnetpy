from dataclasses import dataclass
   
import numpy as np
import scipy.sparse

from statsmodels.genmod.families import family as sm_family
import statsmodels.api as sm

def glmnet_fit(X,
               y,
               weights,
               lambda_val,
               family='Gaussian',
               alpha=1.0,
               intercept=True,
               thresh=1e-7,
               maxit=100000,
               penalty_factor=None, 
               exclude=[],
               lower_limits=-np.inf,
               upper_limits=np.inf,
               warm=None,
               save_fit=False,
               internal_params={'big':1e30},
               from_glmnet_path=False):

    '''An IRLS ....

    .. math::
    
        ???1/2 \sum w_i (y_i - X_i^T \beta)^2 + \sum \lambda \gamma_j [(1-\alpha)/2 \beta^2+\alpha|\beta|]

    over $\beta$, where $\gamma_j$ is the relative penalty factor on the
    j-th variable. If `intercept`, then the term in the first sum is
    $w_i (y_i - \beta_0 - X_i^T \beta)^2$, and we are minimizing over both
    $\beta_0$ and $\beta$.

    None of the inputs are standardized except for `penalty_factor`, which
    is standardized so that they sum up to `nvars`.

    Parameters
    ----------

    X: Union[np.ndarray, scipy.sparse]
        Input matrix, of shape `(nobs, nvars)`; each row is an
        observation vector. If it is a sparse matrix, it is assumed to
        be unstandardized.  If it is not a sparse matrix, it is
        assumed that any standardization needed has already been done.

    y: np.ndarray
        Quantitative response variable.

    weights: np.ndarray
        Observation weights. `elnet_fit` does NOT standardize these weights.

    lambda_val: float
        A single value for the `lambda` hyperparameter.

    alpha: float

        The elasticnet mixing parameter in [0,1].  The penalty is
        defined as $(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.$
        `alpha=1` is the lasso penalty, and `alpha=0` the ridge
        penalty.

    intercept: bool
        Should intercept be fitted (default=`True`) or set to zero (`False`)?

    thresh: float

        Convergence threshold for coordinate descent. Each inner
        coordinate-descent loop continues until the maximum change in the
        objective after any coefficient update is less than thresh times
        the null deviance.  Default value is `1e-7`.

    maxit: int

        Maximum number of passes over the data; default is
        `10^5`.  (If a warm start object is provided, the number
        of passes the warm start object performed is included.)

    penalty_factor: np.ndarray (optional)

        Separate penalty factors can be applied to each
        coefficient. This is a number that multiplies `lambda_val` to
        allow differential shrinkage. Can be 0 for some variables,
        which implies no shrinkage, and that variable is always
        included in the model. Default is 1 for all variables (and
        implicitly infinity for variables listed in `exclude`). Note:
        the penalty factors are internally rescaled to sum to
        `nvars=X.shape[1]`.

    exclude: list

        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.

    lower_limits: Union[List[float, np.ndarray]]

        Vector of lower limits for each coefficient; default
        `-np.inf`. Each of these must be non-positive. Can be
        presented as a single value (which will then be replicated),
        else a vector of length `nvars`.

    upper_limits: Union[List[float, np.ndarray]]

        Vector of upper limits for each coefficient; default
        `np.inf`. See `lower_limits`.

    warm: dict(optional)

        A dict-like with keys `beta` and `a0` containing coefficients
        and intercept respectively which can be used as a warm start.
        For internal use only.

    from_glmnet_path: bool

        Was `glmnet_fit` called from `glmnet_path`?
        Default is `False`. This has implications for computation of the penalty factors.

    save_fit: bool

        Return the warm start object? Default is `False`.

    Returns
    -------

    result: ElnetResult

    '''

    # get the relevant family functions

    if type(family) == type('Gaussian'):
        family = getattr(sm_family, family)
                
    control = {'big':9.9e35,
               'mxitnr':25,
               'epsnr':1e-6}

    nobs, nvars = X.shape

    # if calling from glmnet_path(), we do not need to check on exclude
    # and penalty_factor arguments as they have been prepared by glmnet_path()

    if not from_glmnet_path:

        # check and standardize penalty factors (to sum to nvars)
        _isinf_penalty = np.isinf(penalty_factor)
        if np.any(_isinf_penalty):
            exclude.extend(np.nonzero(_isinf_penalty)[0])
            exclude = np.unique(exclude)

        if exclude.shape[0] > 0:
            if exclude.max() >= nvars:
                raise ValueError("Some excluded variables out of range")
            penalty_factor[exclude] = 1 # now can change penalty_factor

        vp = np.maximum(0, penalty_factor).reshape((-1,1))
        vp = (vp * nvars / vp.sum())

    else:
        vp = np.asarray(penalty_factor, float)

    if lower_limits == -np.inf:
        lower_limits = -np.inf * np.ones(nvars)

    if upper_limits == np.inf:
        upper_limits = np.inf * np.ones(nvars)

    lower_limits = lower_limits[:nvars]
    upper_limits = upper_limits[:nvars]

    if lower_limits.shape[0] < nvars:
        raise ValueError('lower_limits should have shape X.shape[1]')
    if upper_limits.shape[0] < nvars:
        raise ValueError('upper_limits should have shape X.shape[1]')
    lower_limits[lower_limits == -np.inf] = -internal_params['big']
    upper_limits[upper_limits == np.inf] = internal_params['big']

    # computation of null deviance (get mu in the process)

    if not warm:
        start_val = _get_start(X,
                               y,
                               weights,
                               fam_,
                               intercept,
                               is_offset,
                               offset,
                               exclude,
                               vp,
                               alpha)
        
        nulldev = start_val['nulldev']
        mu = start_val['mu']
        fit = None
        coefold = np.zeros(nvars)   # initial coefs = 0
        eta = family.link(mu)
        intold = (eta - offset)[0]

    else:
        if 'warm_fit' in warm:
            fit = warm
            nulldev = fit['nulldev']
            coefold = fit['warm_fit']['a']   # prev value for coefficients
            intold = fit['warm_fit']['aint']    # prev value for intercept
            eta = get_eta(x, coefold, intold)
            mu <- linkinv(eta <- eta + offset)
        } else if (inherits(warm,"list") && "a0" %in% names(warm) &&
                   "beta" %in% names(warm)) {
            nulldev <- _get_start(x,
                                  y,
                                  weights,
                                  family,
                                  intercept,
                                  is_offset,
                                  offset,
                                  exclude,
                                  vp,
                                  alpha)$nulldev
            fit <- warm
            coefold <- fit$beta   # prev value for coefficients
            intold <- fit$a0    # prev value for intercept
            eta <- get_eta(x, coefold, intold)
            mu <- linkinv(eta <- eta + offset)
        } else {
            stop("Invalid warm start object")
        }
    }

    start <- NULL     # current value for coefficients
    start_int <- NULL # current value for intercept
    obj_val_old <- obj_function(y, mu, weights, family, lambda, alpha, coefold, vp)
    if (trace.it == 2) {
        cat("Warm Start Objective:", obj_val_old, fill = TRUE)
    }
    conv <- FALSE      # converged?



    args, nulldev = _elnet_args(X,
                                y,
                                weights,
                                lambda_val,
                                alpha=alpha,
                                intercept=intercept,
                                thresh=thresh,
                                maxit=maxit,
                                penalty_factor=penalty_factor,
                                exclude=exclude,
                                lower_limits=lower_limits,
                                upper_limits=upper_limits,
                                warm=warm,
                                save_fit=save_fit,
                                internal_params=internal_params,
                                from_glmnet_fit=from_glmnet_fit)

    if scipy.sparse.issparse(X):
        wls_fit = sparse_wls(**args)
    else:
        wls_fit = dense_wls(**args)

    nobs, nvars = X.shape

    # if error code > 0, fatal error occurred: stop immediately
    # if error code < 0, non-fatal error occurred: return error code

    if wls_fit['jerr'] != 0:
        errmsg = _jerr_glmnetfit(wls_fit['jerr'], maxit)
        raise ValueError(errmsg['msg'])

    warm_fit = {}
    for key in ["almc", "r", "xv", "ju", "vp",
                "cl", "nx", "a", "aint", "g",
                "ia", "iy", "iz", "mm", "nino",
                "rsqc", "nlp"]:
            warm_fit[key] = wls_fit[key]

    warm_fit['m'] = args['m'] # isn't this always 1?
    warm_fit['no'] = nobs
    warm_fit['ni'] = nvars

    beta = scipy.sparse.csc_array(wls_fit['a']) # shape=(1, nvars)

    out = ElnetResult(a0=wls_fit['aint'],
                      beta=beta,
                      df=np.sum(np.abs(beta) > 0),
                      dim=beta.shape,
                      lambda_val=lambda_val,
                      dev_ratio=wls_fit['rsqc'],
                      nulldev=nulldev,
                      npasses=wls_fit['nlp'],
                      jerr=wls_fit['jerr'],
                      nobs=nobs,
                      warm_fit=warm_fit)

    if not save_fit:
        out.warm_fit = {}

    return out




    # IRLS loop
    for (iter in 1L:control$mxitnr) {
        # some checks for NAs/zeros
        varmu <- variance(mu)
        if (anyNA(varmu)) stop("NAs in V(mu)")
        if (any(varmu == 0)) stop("0s in V(mu)")
        mu.eta.val <- mu.eta(eta)
        if (any(is.na(mu.eta.val))) stop("NAs in d(mu)/d(eta)")

        # compute working response and weights
        z <- (eta - offset) + (y - mu)/mu.eta.val
        w <- (weights * mu.eta.val^2)/variance(mu)

        # have to update the weighted residual in our fit object
        # (in theory g and iy should be updated too, but we actually recompute g
        # and iy anyway in wls.f)
        if (!is.null(fit)) {
            fit$warm_fit$r <- w * (z - eta + offset)
        }

        # do WLS with warmstart from previous iteration
        fit <- elnet.fit(x, z, w, lambda, alpha, intercept,
                         thresh = thresh, maxit = maxit, penalty.factor = vp,
                         exclude = exclude, lower.limits = lower.limits,
                         upper.limits = upper.limits, warm = fit,
                         from.glmnet.fit = TRUE, save.fit = TRUE)
        if (fit$jerr != 0) return(list(jerr = fit$jerr))

        # update coefficients, eta, mu and obj_val
        start <- fit$warm_fit$a
        start_int <- fit$warm_fit$aint
        eta <- get_eta(x, start, start_int)
        mu <- linkinv(eta <- eta + offset)
        obj_val <- obj_function(y, mu, weights, family, lambda, alpha, start, vp)
        if (trace.it == 2) cat("Iteration", iter, "Objective:", obj_val, fill = TRUE)

        boundary <- FALSE
        halved <- FALSE  # did we have to halve the step size?
        # if objective function is not finite, keep halving the stepsize until it is finite
        # for the halving step, we probably have to adjust fit$g as well?
        if (!is.finite(obj_val) || obj_val > control$big) {
            warning("Infinite objective function!", call. = FALSE)
            if (is.null(coefold) || is.null(intold))
                stop("no valid set of coefficients has been found: please supply starting values",
                     call. = FALSE)
            warning("step size truncated due to divergence", call. = FALSE)
            ii <- 1
            while (!is.finite(obj_val) || obj_val > control$big) {
                if (ii > control$mxitnr)
                    stop("inner loop 1; cannot correct step size", call. = FALSE)
                ii <- ii + 1
                start <- (start + coefold)/2
                start_int <- (start_int + intold)/2
                eta <- get_eta(x, start, start_int)
                mu <- linkinv(eta <- eta + offset)
                obj_val <- obj_function(y, mu, weights, family, lambda, alpha, start, vp)
                if (trace.it == 2) cat("Iteration", iter, " Halved step 1, Objective:",
                               obj_val, fill = TRUE)
            }
            boundary <- TRUE
            halved <- TRUE
        }
        # if some of the new eta or mu are invalid, keep halving stepsize until valid
        if (!(valideta(eta) && validmu(mu))) {
            warning("Invalid eta/mu!", call. = FALSE)
            if (is.null(coefold) || is.null(intold))
                stop("no valid set of coefficients has been found: please supply starting values",
                     call. = FALSE)
            warning("step size truncated: out of bounds", call. = FALSE)
            ii <- 1
            while (!(valideta(eta) && validmu(mu))) {
                if (ii > control$mxitnr)
                    stop("inner loop 2; cannot correct step size", call. = FALSE)
                ii <- ii + 1
                start <- (start + coefold)/2
                start_int <- (start_int + intold)/2
                eta <- get_eta(x, start, start_int)
                mu <- linkinv(eta <- eta + offset)
            }
            boundary <- TRUE
            halved <- TRUE
            obj_val <- obj_function(y, mu, weights, family, lambda, alpha, start, vp)
            if (trace.it == 2) cat("Iteration", iter, " Halved step 2, Objective:", obj_val, fill = TRUE)
        }
        # extra halving step if objective function value actually increased
        if (obj_val > obj_val_old + 1e-7) {
            ii <- 1
            while (obj_val > obj_val_old + 1e-7) {
                if (ii > control$mxitnr)
                    stop("inner loop 3; cannot correct step size", call. = FALSE)
                ii <- ii + 1
                start <- (start + coefold)/2
                start_int <- (start_int + intold)/2
                eta <- get_eta(x, start, start_int)
                mu <- linkinv(eta <- eta + offset)
                obj_val <- obj_function(y, mu, weights, family, lambda, alpha, start, vp)
                if (trace.it == 2) cat("Iteration", iter, " Halved step 3, Objective:",
                                       obj_val, fill = TRUE)
            }
            halved <- TRUE
        }

        # if we did any halving, we have to update the coefficients, intercept
        # and weighted residual in the warm_fit object
        if (halved) {
            fit$warm_fit$a <- start
            fit$warm_fit$aint <- start_int
            fit$warm_fit$r <- w * (z - eta)
        }

        # test for convergence
        if (abs(obj_val - obj_val_old)/(0.1 + abs(obj_val)) < control$epsnr) {
            conv <- TRUE
            break
        }
        else {
            coefold <- start
            intold <- start_int
            obj_val_old <- obj_val
        }
    }
    # end of IRLS loop

    # checks on convergence and fitted values
    if (!conv)
        warning("glmnet.fit: algorithm did not converge", call. = FALSE)
    if (boundary)
        warning("glmnet.fit: algorithm stopped at boundary value", call. = FALSE)

    # some extra warnings, printed only if trace.it == 2
    if (trace.it == 2) {
        eps <- 10 * .Machine$double.eps
        if ((family$family == "binomial") && (any(mu > 1 - eps) || any(mu < eps)))
                warning("glm.fit: fitted probabilities numerically 0 or 1 occurred",
                        call. = FALSE)
        if ((family$family == "poisson") && (any(mu < eps)))
                warning("glm.fit: fitted rates numerically 0 occurred",
                        call. = FALSE)
    }

    # prepare output object
    if (save.fit == FALSE) {
        fit$warm_fit <- NULL
    }
    # overwrite values from elnet.fit object
    fit$call <- this.call
    fit$offset <- is.offset
    fit$nulldev <- nulldev
    fit$dev.ratio <- 1 - dev_function(y, mu, weights, family) / nulldev

    # add new key-value pairs to list
    fit$family <- family
    fit$converged <- conv
    fit$boundary <- boundary
    fit$obj_function <- obj_val

    class(fit) <- c("glmnetfit", "glmnet")
    fit
}

#' Get null deviance, starting mu and lambda max
#'
#' Return the null deviance, starting mu and lambda max values for
#' initialization. For internal use only.
#'
#' This function is called by \code{glmnet.path} for null deviance, starting mu
#' and lambda max values. It is also called by \code{glmnet.fit} when used
#' without warmstart, but they only use the null deviance and starting mu values.
#'
#' When \code{x} is not sparse, it is expected to already by centered and scaled.
#' When \code{x} is sparse, the function will get its attributes \code{xm} and
#' \code{xs} for its centering and scaling factors.
#'
#' Note that whether \code{x} is centered & scaled or not, the values of \code{mu}
#' and \code{nulldev} don't change. However, the value of \code{lambda_max} does
#' change, and we need \code{xm} and \code{xs} to get the correct value.
#'
#' @param x Input matrix, of dimension \code{nobs x nvars}; each row is an
#' observation vector. If it is a sparse matrix, it is assumed to be unstandardized.
#' It should have attributes \code{xm} and \code{xs}, where \code{xm(j)} and
#' \code{xs(j)} are the centering and scaling factors for variable j respsectively.
#' If it is not a sparse matrix, it is assumed to be standardized.
#' @param y Quantitative response variable.
#' @param weights Observation weights.
#' @param family A description of the error distribution and link function to be
#' used in the model. This is the result of a call to a family function.
#' (See \code{\link[stats:family]{family}} for details on family functions.)
#' @param intercept Does the model we are fitting have an intercept term or not?
#' @param is.offset Is the model being fit with an offset or not?
#' @param offset Offset for the model. If \code{is.offset=FALSE}, this should be
#' a zero vector of the same length as \code{y}.
#' @param exclude Indices of variables to be excluded from the model.
#' @param vp Separate penalty factors can be applied to each coefficient.
#' @param alpha The elasticnet mixing parameter, with \eqn{0 \le \alpha \le 1}.

def _get_start(X,
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
        if offset:
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

    nulldev = dev_function(y, mu, weights, family)

    # if some penalty factors are zero, we have to recompute mu

    vp_zero = sorted(set(exclude).difference(np.nonzero(vp == 0)[0]))
    if vp_zero:
        tempX = X[, vp_zero]

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

        xm = X.T @ weights
        xm2 = (X*X).T @ weights
        xs = xm2 - xm**2

        g = np.abs(t(X) @ rv - np.sum(rv) * xm / xs)

    else:
        g = np.abs(t(X) @ rv)

    g = g * ju / (vp + (vp <= 0))
    lambda_max = np.max(g) / max(alpha, 1e-3)

    return {'nulldev':nulldev,
            'mu':mu,
            'lambda_max':lambda_max}
}

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
obj_function <- function(y, mu, weights, family,
                         lambda, alpha, coefficients, vp) {
    dev_function(y, mu, weights, family) / 2 +
        lambda * pen_function(coefficients, alpha, vp)
}

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
pen_function <- function(coefficients, alpha = 1.0, vp = 1.0) {
    sum(vp * (alpha * abs(coefficients) + (1-alpha)/2 * coefficients^2))
}

#' Elastic net deviance value
#'
#' Returns the elastic net deviance value.
#'
#' @param y Quantitative response variable.
#' @param mu Model's predictions for \code{y}.
#' @param weights Observation weights.
#' @param family A description of the error distribution and link functio
