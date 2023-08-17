import warnings
from dataclasses import dataclass
   
import numpy as np
import scipy.sparse

from statsmodels.genmod.families import family as sm_family
import statsmodels.api as sm

from ._utils import (_get_limits,
                     _get_vp,
                     _get_eta,
                     _jerr_elnetfit,
                     _obj_function,
                     _get_start,
                     _dev_function)
from .elnet_fit import elnet_fit, ElNetResult

@dataclass
class GLMNetResult(ElNetResult):

    family: sm_family.Family
    offset: bool
    converged: bool
    boundary: bool
    obj_function: float

@dataclass
class glmnet_state(object):

    mu: np.ndarray
    eta: np.ndarray
    obj_val: float
    obj_val_old: float
    
def glmnet_fit(X,
               y,
               weights,
               lambda_val,
               family='Gaussian',
               link=None,
               offset=None,
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
        F = getattr(sm_family, family)
        family = F(link=link)
        # FIX THESE 
        valideta = lambda eta: True
        validmu = lambda mu: True
        
    control = {'big':9.9e35,
               'mxitnr':25,
               'epsnr':1e-6}

    nobs, nvars = X.shape

    # get offset

    is_offset = offset is not None
    if not offset:
        offset = 0 * y

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
                               family,
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

    state = glmnet_state(mu=mu,
                         eta=eta,
                         obj_val=np.inf,
                         obj_val_old=obj_val_old)

    converged = False

    for iter in range(control['mxitnr']):

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        # compute working response and weights
        z = (state.eta - offset) + (y - state.mu) / dmu_deta
        w = (weights * dmu_deta**2)/varmu

        # have to update the weighted residual in our fit object
        # (in theory g and iy should be updated too, but we actually recompute g
        # and iy anyway in wls.f)
        if fit is not None:
            fit['warm_fit']['r'] = w * (z - state.eta + offset)

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

        if fit.jerr != 0:
            errmsg = _jerr_elnetfit(fit.jerr, maxit)
            raise ValueError(errmsg['msg'])

        # update coefficients, eta, mu and obj_val

        start = fit.warm_fit['a']
        start_int = fit.warm_fit['aint']
        state.eta = _get_eta(X,
                             X_scale,
                             start,
                             start_int)

        state.mu = family.link.inverse(eta + offset)
        state.obj_val = _obj_function(y,
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

        # three checks we'll apply

        def finite_objective(state):
            boundary = True
            halved = True
            return np.isfinite(state.obj_val) and state.obj_val < control['big'], boundary, halved

        def valid(state):
            boundary = True
            halved = True
            return valideta(state.eta) and validmu(state.mu), boundary, halved
            
        def decreased_obj(state):
            boundary = False
            halved = True
            return state.obj_val < state.obj_val_old + 1e-7, boundary, halved

        for test, msg in [(finite_objective,
                           "Non finite objective function! Step size truncated due to divergence."),
                          (valid,
                           "Invalid eta/mu! Step size truncated: out of bounds."),
                          (decreased_obj,
                           "foo")]:

            if not test(state)[0]:
                warnings.warn(msg)
                if np.any(np.isnan(coefold)) or np.isnan(intold):
                    raise ValueError("No valid set of coefficients has been found: please supply starting values")

                ii = 1
                check, boundary_, halved_ = test(state)
                boundary = boundary or boundary_
                halved = halved or halved_
                
                while not check:
                    if ii > control['mxitnr']:
                        raise ValueError(f"inner loop {test}; cannot correct step size")
                    ii += 1
                    start = (start + coefold)/2
                    start_int = (start_int + intold)/2

                    state.eta = _get_eta(X,
                                         X_scale,
                                         start,
                                         start_int)

                    state.mu = family.link.inverse(state.eta + offset)
                    state.obj_val = _obj_function(y,
                                                  state.mu,
                                                  weights,
                                                  family,
                                                  lambda_val,
                                                  alpha,
                                                  start,
                                                  vp)
                    check, boundary_, halved_ = test(state)

        # if we did any halving, we have to update the coefficients, intercept
        # and weighted residual in the warm_fit object
        if halved:
            fit.warm_fit['a'] = start
            fit.warm_fit['aint'] = start_int
            fit.warm_fit['r'] =  w * (z - state.eta)

        # test for convergence
        if (np.fabs(state.obj_val - state.obj_val_old)/(0.1 + abs(state.obj_val)) < control['epsnr']):
            converged = True
            break

        else:
            coefold = start
            intold = start_int
            state.obj_val_old = obj_val

    # end of IRLS loop

    # checks on convergence and fitted values
    if not converged:
        warnings.warn("glmnet_fit: algorithm did not converge")
    if boundary:
        warnings.warn("glmnet_fit: algorithm stopped at boundary value")

    # # some extra warnings, printed only if trace.it == 2
    # if (trace.it == 2) {
    #     eps <- 10 * .Machine$double.eps
    #     if ((family$family == "binomial") && (any(mu > 1 - eps) || any(mu < eps)))
    #             warning("glm.fit: fitted probabilities numerically 0 or 1 occurred",
    #                     call. = FALSE)
    #     if ((family$family == "poisson") && (any(mu < eps)))
    #             warning("glm.fit: fitted rates numerically 0 occurred",
    #                     call. = FALSE)
    # }

    # prepare output object
    if not save_fit:
        fit.warm_fit = {}

    # create a GLMNetResult

    args = fit.__dict__
    args['offset'] = is_offset
    args['nulldev'] = nulldev
    args['dev_ratio'] = 1 - _dev_function(y, state.mu, weights, family) / nulldev
    args['family'] = family
    args['converged'] = converged
    args['boundary'] = boundary
    args['obj_function'] = state.obj_val

    return GLMNetResult(**args)

