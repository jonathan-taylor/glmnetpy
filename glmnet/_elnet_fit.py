#' WARNING: Users should not call \code{elnet.fit} directly. Higher-level functions
#' in this package call \code{elnet.fit} as a subroutine. If a warm start object
#' is provided, some of the other arguments in the function may be overriden.
#'
#' \code{elnet.fit} is essentially a wrapper around a C++ subroutine which
#' minimizes
#'
#' \deqn{1/2 \sum w_i (y_i - X_i^T \beta)^2 + \sum \lambda \gamma_j
#' [(1-\alpha)/2 \beta^2+\alpha|\beta|],}
#'
#' over \eqn{\beta}, where \eqn{\gamma_j} is the relative penalty factor on the
#' jth variable. If \code{intercept = TRUE}, then the term in the first sum is
#' \eqn{w_i (y_i - \beta_0 - X_i^T \beta)^2}, and we are minimizing over both
#' \eqn{\beta_0} and \eqn{\beta}.
#'
#' None of the inputs are standardized except for \code{penalty.factor}, which
#' is standardized so that they sum up to \code{nvars}.
#'
#' @param x Input matrix, of dimension \code{nobs x nvars}; each row is an
#' observation vector. If it is a sparse matrix, it is assumed to be unstandardized.
#' It should have attributes \code{xm} and \code{xs}, where \code{xm(j)} and
#' \code{xs(j)} are the centering and scaling factors for variable j respsectively.
#' If it is not a sparse matrix, it is assumed that any standardization needed
#' has already been done.
#' @param y Quantitative response variable.
#' @param weights Observation weights. \code{elnet.fit} does NOT standardize
#' these weights.
#' @param lambda A single value for the \code{lambda} hyperparameter.
#' @param alpha The elasticnet mixing parameter, with \eqn{0 \le \alpha \le 1}.
#' The penalty is defined as \deqn{(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.}
#' \code{alpha=1} is the lasso penalty, and \code{alpha=0} the ridge penalty.
#' @param intercept Should intercept be fitted (default=TRUE) or set to zero (FALSE)?
#' @param thresh Convergence threshold for coordinate descent. Each inner
#' coordinate-descent loop continues until the maximum change in the objective
#' after any coefficient update is less than thresh times the null deviance.
#' Default value is \code{1e-7}.
#' @param maxit Maximum number of passes over the data; default is \code{10^5}.
#' (If a warm start object is provided, the number of passes the warm start object
#' performed is included.)
#' @param penalty.factor Separate penalty factors can be applied to each
#' coefficient. This is a number that multiplies \code{lambda} to allow differential
#' shrinkage. Can be 0 for some variables, which implies no shrinkage, and that
#' variable is always included in the model. Default is 1 for all variables (and
#' implicitly infinity for variables listed in exclude). Note: the penalty
#' factors are internally rescaled to sum to \code{nvars}.
#' @param exclude Indices of variables to be excluded from the model. Default is
#' none. Equivalent to an infinite penalty factor.
#' @param lower.limits Vector of lower limits for each coefficient; default
#' \code{-Inf}. Each of these must be non-positive. Can be presented as a single
#' value (which will then be replicated), else a vector of length \code{nvars}.
#' @param upper.limits Vector of upper limits for each coefficient; default
#' \code{Inf}. See \code{lower.limits}.
#' @param warm Either a \code{glmnetfit} object or a list (with names \code{beta}
#' and \code{a0} containing coefficients and intercept respectively) which can
#' be used as a warm start. Default is \code{NULL}, indicating no warm start.
#' For internal use only.
#' @param from.glmnet.fit Was \code{elnet.fit()} called from \code{glmnet.fit()}?
#' Default is FALSE.This has implications for computation of the penalty factors.
#' @param save.fit Return the warm start object? Default is FALSE.
#'
#' @return An object with class "glmnetfit" and "glmnet". The list returned has
#' the same keys as that of a \code{glmnet} object, except that it might have an
#' additional \code{warm_fit} key.
#' \item{a0}{Intercept value.}
#' \item{beta}{A \code{nvars x 1} matrix of coefficients, stored in sparse matrix
#' format.}
#' \item{df}{The number of nonzero coefficients.}
#' \item{dim}{Dimension of coefficient matrix.}
#' \item{lambda}{Lambda value used.}
#' \item{dev.ratio}{The fraction of (null) deviance explained. The deviance
#' calculations incorporate weights if present in the model. The deviance is
#' defined to be 2*(loglike_sat - loglike), where loglike_sat is the log-likelihood
#' for the saturated model (a model with a free parameter per observation).
#' Hence dev.ratio=1-dev/nulldev.}
#' \item{nulldev}{Null deviance (per observation). This is defined to be
#' 2*(loglike_sat -loglike(Null)). The null model refers to the intercept model.}
#' \item{npasses}{Total passes over the data.}
#' \item{jerr}{Error flag, for warnings and errors (largely for internal
#' debugging).}
#' \item{offset}{Always FALSE, since offsets do not appear in the WLS problem.
#' Included for compability with glmnet output.}
#' \item{call}{The call that produced this object.}
#' \item{nobs}{Number of observations.}
#' \item{warm_fit}{If \code{save.fit=TRUE}, output of C++ routine, used for
#' warm star

import numpy as np
import scipy.sparse

from .glmnetpp import wls as dense_wls
from .glmnetpp import spwls as sparse_wls

def _elnet_fit(X,
               y,
               weights,
               lambda_val,
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
               internal_params={'big':1e10},
               from_glmnet_fit=False):
    
    if scipy.sparse.issparse(X):
        X = X.tocsc()

    exclude = np.asarray(exclude, int)

    nobs, nvars = X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # compute null deviance
    weights = weights / weights.sum()
    
    ybar = np.sum(y * weights) / np.sum(weights)
    nulldev = np.sum(weights * (y - ybar)**2)

    # if class "glmnetfit" warmstart object provided, pull whatever we want out of it
    # else, prepare arguments, then check if coefs provided as warmstart
    # (if only coefs are given as warmstart, we prepare the other arguments
    # as if no warmstart was provided)

    if warm is not None: # assumes it is a dictionary like `warm_fit`
        a = warm['a']
        aint = warm['aint']
        alm0 = warm['almc']
        cl = warm['cl']
        g = warm['g'].reshape((-1,1))
        ia = warm['ia']
        iy = warm['iy']
        iz = warm['iz']
        ju = warm['ju'].reshape((-1,1))
        m = warm['m']
        mm = warm['mm'].reshape((-1,1))
        nino = warm['nino']
        nobs = warm['no']
        nvars = warm['ni']
        nlp = warm['nlp']
        nx = warm['nx']
        r = warm['r'].reshape((-1,1))
        rsqc = warm['rsqc']
        xv = warm['xv'].reshape((-1,1))
        vp = warm['vp'].reshape((-1,1))
    else:
        
        # if calling from glmnet.fit(), we do not need to check on exclude
        # and penalty.factor arguments as they have been prepared by glmnet.fit()
        # Also exclude will include variance 0 columns
        if not from_glmnet_fit:

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

        # compute ju
        # assume that there are no constant variables
        ju = np.ones((nvars, 1), int)
        ju[exclude] = 0

        # compute cl from upper and lower limits

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
        
        cl = np.asarray([lower_limits,
                         upper_limits], float)

        nx = nvars #  as.integer(nvars)

                                         # From elnet.fit R code
        a  = np.zeros((nvars, 1))        # double(nvars)
        aint = 0.                        # double(1) -- mismatch?
        alm0  = 0.                       # double(1) -- mismatch?
        g = np.zeros((nvars, 1))         # double(nvars) -- mismatch?
        ia = np.zeros((nx, 1), int)      # integer(nx)
        iy = np.zeros((nvars, 1), int)   # integer(nvars)     
        iz = 0                           # integer(1) -- mismatch?
        m = 1                            # as.integer(1)
        mm = np.zeros((nvars, 1), int)   # integer(nvars) -- mismatch?
        nino = int(0)                    # integer(1)
        nlp = 0                          # integer(1) -- mismatch?
        r =  (weights * y).reshape((-1,1))
        rsqc = 0.                        # double(1) -- mismatch?
        xv = np.zeros((nvars, 1))        # double(nvars)

        # check if coefs were provided as warmstart: if so, use them

        if warm is not None:
            if 'beta' in warm and 'a0' in warm:
                a = np.asarray(warm['beta'], float)
                aint = np.asarray(warm['a0'], float)
                mu = X @ a + aint
                r = (weights * (y - mu)).reshape((-1,1))
                rsqc = 1 - np.sum(weights * (y - mu)**2) / nulldev
            else:
                raise ValueError('warm start object should have "beta" and "a0"')

    # for the parameters here, we are overriding the values provided by the
    # warmstart object

    alpha = float(alpha)                            # as.double(alpha)
    almc = float(lambda_val)                        # as.double(lambda)
    intr = int(intercept)                           # as.integer(intercept)
    jerr = 0                                        # integer(1) -- mismatch?
    maxit = int(maxit)                              # as.integer(maxit)
    thr = float(thresh)                             # as.double(thresh)
    v = np.asarray(weights, float).reshape((-1,1))  # as.double(weights)

    a_new = a.copy()

    # take out components of x and run C++ subroutine
    if scipy.sparse.issparse(X):

        xm = X.T @ weights
        xm2 = (X*X) @ weights
        xs = xm2 - xm**2
        
        data_array = X.data
        indices_array = X.indices
        indptr_array = X.indptr

        wls_fit = sparse_wls(alm0=alm0,
                             almc=almc,
                             alpha=alpha,
                             m=m,
                             no=nobs,
                             ni=nvars,
                             x=X,
                             xm=xm,
                             xs=xs,
                             r=r,
                             xv=xv,
                             v=v,
                             intr=intr,
                             ju=ju,
                             vp=vp,
                             cl=cl,
                             nx=nx,
                             thr=thr,
                             maxit=maxit,
                             a=a_new,
                             aint=aint,
                             g=g,
                             ia=ia,
                             iy=iy,
                             iz=iz,
                             mm=mm,
                             nino=nino,
                             rsqc=rsqc,
                             nlp=nlp,
                             jerr=jerr)
    else:
        wls_fit = dense_wls(alm0=alm0,
                            almc=almc,
                            alpha=alpha,
                            m=m,
                            no=nobs,
                            ni=nvars,
                            x=X,
                            r=r,
                            xv=xv,
                            v=v,
                            intr=intr,
                            ju=ju,
                            vp=vp,
                            cl=cl,
                            nx=nx,
                            thr=thr,
                            maxit=maxit,
                            a=a_new,
                            aint=aint,
                            g=g,
                            ia=ia,
                            iy=iy,
                            iz=iz,
                            mm=mm,
                            nino=nino,
                            rsqc=rsqc,
                            nlp=nlp,
                            jerr=jerr)


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

    warm_fit['m'] = m
    warm_fit['no'] = nobs
    warm_fit['ni'] = nvars

    beta = scipy.sparse.csc_array(wls_fit['a']) # shape=(1, nvars)

    out = {'a0':wls_fit['aint'],
           'beta':beta,
           'df':np.sum(np.abs(beta) > 0),
           'dim':beta.shape,
           'lambda_val':lambda_val,
           'dev.ratio':wls_fit['rsqc'],
           'nulldev':nulldev,
           'npasses':wls_fit['nlp'],
           'jerr':wls_fit['jerr'],
           'offset':False,
           'nobs':nobs,
           'warm_fit':warm_fit}
    if not save_fit:
        del(out['warm_fit'])

    return out

def _jerr_glmnetfit(n, maxit, k=None):
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