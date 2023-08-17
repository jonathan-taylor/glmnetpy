from dataclasses import dataclass
   
import numpy as np
import scipy.sparse

from statsmodels.genmod.families import family as sm_family
import statsmodels.api as sm

from ._utils import (_get_limits,
                     _get_vp,
                     _get_eta,
                     _jerr_elnetfit,
                     _obj_function)

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

    result: GLMnetResult

    '''

    # get the relevant family functions

    if type(family) == str:
        family = getattr(sm_family, family)
                
    control = {'big':9.9e35,
               'mxitnr':25,
               'epsnr':1e-6}

    nobs, nvars = X.shape

    # if calling from glmnet_path(), we do not need to check on exclude
    # and penalty_factor arguments as they have been prepared by glmnet_path()

    if not from_glmnet_path:
        vp, exclude = _get_vp(penalty_factor,
                              exclude,
                              nvars)
    else:
        vp = np.asarray(penalty_factor, float)

    lower_limits, upper_limits = _get_limits(lower_limits,
                                             upper_limits,
                                             nvars,
                                             internal_params['big'])


    # computation of null deviance (get mu in the process)

    if scipy.sparse.issparse(X):
        X = X.tocsc()
        xm = X.T @ weights
        xm2 = (X*X).T @ weights
        xs = xm2 - xm**2
        X_scale = (xm, xs)
    else:
        X_scale = None

    if not warm:
        start_val = _get_start(X,
                               X_scale,
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
        fit = warm
        if 'warm_fit' in warm:
            nulldev = fit['nulldev']
            coefold = fit['warm_fit']['a']   # prev value for coefficients
            intold = fit['warm_fit']['aint']    # prev value for intercept
            eta = _get_eta(X,
                           X_scale,
                           coefold,
                           intold)
            mu = family.link.inverse(eta + offset)

        elif 'a0' in warm and 'beta' in warm:
            nulldev = _get_start(X,
                                 X_scale,
                                 y,
                                 weights,
                                 family,
                                 intercept,
                                 is_offset,
                                 offset,
                                 exclude,
                                 vp,
                                 alpha)['nulldev']

            coefold = warm['beta']   # prev value for coefficients
            intold = warm['a0']      # prev value for intercept
            eta = _get_eta(X,
                           X_scale,
                           coefold,
                           intold)
            mu = family.link.inverse(eta + offset)
        else:
            raise ValueError("Invalid warm start object")

    # IRLS 

    start = None     # current value for coefficients
    start_int = None # current value for intercept

    obj_val_old = _obj_function(y,
                                mu,
                                weights,
                                family,
                                lambda_val,
                                alpha,
                                coefold,
                                vp)

    conv = False      # converged?

    for iter in range(control['mxitnr']):

        # some checks for NAs/zeros
        varmu = family.variance(mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        # compute working response and weights
        z = (eta - offset) + (y - mu) / dmu_deta
        w = (weights * dmu_deta**2)/varmu

        # have to update the weighted residual in our fit object
        # (in theory g and iy should be updated too, but we actually recompute g
        # and iy anyway in wls.f)
        if fit is not None:
            fit['warm_fit']['r'] = w * (z - eta + offset)

        # do WLS with warmstart from previous iteration
        fit = elnet_fit(X=X,
                        y=z,
                        weights=w,
                        lambda_val=lambda_val,
                        alpha=alpha,
                        intercept=intercept,
                        thresh=thresh,
                        maxit=maxit,
                        penalty_factor=vp,
                        exclude=exclude,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        warm=fit,
                        from_glmnet_fit=True,
                        save_fit=True)

        if fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(fit['jerr'], maxit)
            raise ValueError(errmsg['msg'])

        # update coefficients, eta, mu and obj_val
        start = fit['warm_fit']['a']
        start_int = fit['warm_fit']['aint']
        eta = _get_eta(X,
                       X_scale,
                       start,
                       start_int)

        mu = family.link.inverse(eta + offset)
        obj_val = _obj_function(y,
                                mu,
                                weights,
                                family,
                                lambda_val,
                                alpha,
                                start,
                                vp)

        boundary = False
        halved = False  # did we have to halve the step size?
        # if objective function is not finite, keep halving the stepsize until it is finite
        # for the halving step, we probably have to adjust fit$g as well?
        if not np.isfinite(obj_val) or obj_val > control['big']:
            warnings.warn("Non finite objective function!")
            if np.any(np.isnan(coefold)) or np.isnan(intold):
                raise ValueError("no valid set of coefficients has been found: please supply starting values")

            warnings.warn("step size truncated due to divergence")

            ii = 1
            while (not np.isfinite(obj_val) or obj_val > control['big']):
                if ii > control['mxitnr']:
                    raise ValueError("inner loop 1; cannot correct step size")
                ii += 1
                start = (start + coefold)/2
                start_int = (start_int + intold)/2

                eta = _get_eta(X,
                               X_scale,
                               start,
                               start_int)

                mu = family.link.inverse(eta + offset)
                obj_val = _obj_function(y,
                                        mu,
                                        weights,
                                        family,
                                        lambda_val,
                                        alpha,
                                        start,
                                        vp)
            boundary = True
            halved = True

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
        xm, xs = X_scale
        g = np.abs(t(X) @ rv - np.sum(rv) * xm / xs)
    else:
        g = np.abs(t(X) @ rv)

    g = g * ju / (vp + (vp <= 0))
    lambda_max = np.max(g) / max(alpha, 1e-3)

    return {'nulldev':nulldev,
            'mu':mu,
            'lambda_max':lambda_max}


