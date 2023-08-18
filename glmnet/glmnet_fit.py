import warnings
from dataclasses import dataclass, asdict, field
   
import numpy as np
import scipy.sparse

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links
import statsmodels.api as sm

from ._utils import (_jerr_elnetfit,
                     _obj_function,
                     _dev_function)

from .elnet_fit import (elnet_fit,
                        ElNetResult,
                        DesignSpec,
                        ElNetSpec,
                        ElNetControl)

@dataclass
class GLMNetControl(ElNetControl):

    mxitnr: int = 25
    epsnr: float = 1e-6

@dataclass
class GLMNetSpec(ElNetSpec):

    offset: np.ndarray = None
    family: sm_family.Family = field(default_factory=sm_family.Gaussian)
    control: GLMNetControl = field(default_factory=GLMNetControl)

    def __post_init__(self):
        ElNetSpec.__post_init__(self)
        if self.offset is None:
            self.is_offset = False
            self.offset = np.zeros(self.y.shape)
        else:
            self.is_offset = True

    def get_start(self):
        (design,
         y,
         weights,
         family,
         intercept,
         is_offset,
         offset,
         exclude,
         vp,
         alpha) = (self.design,
                   self.y,
                   self.weights,
                   self.family,
                   self.intercept,
                   self.is_offset,
                   self.offset,
                   self.exclude,
                   self.vp,
                   self.alpha)

        X = design.X
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
            xm, xs = design.xm, design.xs
            g = np.abs((X.T @ rv - np.sum(rv) * xm) / xs)
        else:
            g = np.abs(X.T @ rv)

        g = g * ju / (vp + (vp <= 0))
        lambda_max = np.max(g) / max(alpha, 1e-3)
        print('lambda_max', lambda_max)
        return {'nulldev':nulldev,
                'mu':mu,
                'lambda_max':lambda_max}

    def obj_function(self,
                     mu,
                     coefficients):

        return _obj_function(self.y,
                             mu,
                             self.weights,
                             self.family,
                             self.lambda_val,
                             self.alpha,
                             coefficients,
                             self.vp)

    def dev_function(self,
                     mu):

        return _dev_function(self.y,
                             mu,
                             self.weights,
                             self.family)

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

    warm: dict

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
        if link is not None:
            L = getattr(sm_links, link)()
            family = F(L)
        else:
            family = F()
        # FIX THESE 
        valideta = lambda eta: True
        validmu = lambda mu: True
        
    control = GLMNetControl(thresh=thresh,
                            maxit=maxit)
    
    problem = GLMNetSpec(X=X,
                         y=y,
                         offset=offset,
                         weights=weights,
                         family=family,
                         lambda_val=lambda_val,
                         alpha=alpha,
                         intercept=intercept,
                         penalty_factor=penalty_factor,
                         lower_limits=lower_limits,
                         upper_limits=upper_limits,
                         exclude=exclude,
                         control=control)

    nobs, nvars = problem.X.shape

    # get offset

    is_offset = offset is not None

    # if calling from glmnet_path(), we do not need to check on exclude
    # and penalty_factor arguments as they have been prepared by glmnet_path()

    # computation of null deviance (get mu in the process)

    design = DesignSpec(X, weights)

    if not warm:
        start_val = problem.get_start()
        
        nulldev = start_val['nulldev']
        mu = start_val['mu']
        fit = None
        coefold = np.zeros(nvars)   # initial coefs = 0
        eta = family.link(mu)
        intold = (eta - problem.offset)[0]

    else:
        fit = warm
        if 'warm_fit' in warm:
            nulldev = fit['nulldev']
            coefold = fit.warm_fit.a   # prev value for coefficients
            intold = fit.warm_fit.aint    # prev value for intercept
            eta = design.get_eta(coefold,
                                 intold)
            mu = family.link.inverse(eta + problem.offset)

        elif 'a0' in warm and 'beta' in warm:
            nulldev = problem.get_start()['nulldev']

            coefold = warm['beta']   # prev value for coefficients
            intold = warm['a0']      # prev value for intercept
            eta = design.get_eta(coefold,
                                 intold)
            mu = family.link.inverse(eta + problem.offset)
        else:
            raise ValueError("Invalid warm start object")

    # IRLS 

    start = None     # current value for coefficients
    start_int = None # current value for intercept

    obj_val_old = problem.obj_function(mu,
                                       coefold)
    print(coefold, obj_val_old, 'huh')
    
    state = glmnet_state(mu=mu,
                         eta=eta,
                         obj_val=np.inf,
                         obj_val_old=obj_val_old)

    converged = False

    for iter in range(problem.control.mxitnr):

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        # compute working response and weights
        z = (state.eta - problem.offset) + (y - state.mu) / dmu_deta
        w = (weights * dmu_deta**2)/varmu

        # have to update the weighted residual in our fit object
        # (in theory g and iy should be updated too, but we actually recompute g
        # and iy anyway in wls.f)
        if fit is not None:
            fit.warm_fit.r = w * (z - state.eta + problem.offset)
            warm = fit.warm_fit
        else:
            warm = None

        # do WLS with warmstart from previous iteration
        fit = elnet_fit(X=X,
                        y=z,
                        weights=w,
                        lambda_val=lambda_val,
                        alpha=alpha,
                        intercept=intercept,
                        thresh=thresh,
                        maxit=maxit,
                        penalty_factor=problem.vp,
                        exclude=exclude,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        warm=warm,
                        from_glmnet_fit=True,
                        save_fit=True)

        if fit.jerr != 0:
            errmsg = _jerr_elnetfit(fit.jerr, maxit)
            raise ValueError(errmsg['msg'])

        # update coefficients, eta, mu and obj_val

        start = fit.warm_fit.a
        start_int = fit.warm_fit.aint
        state.eta = design.get_eta(start,
                                   start_int)

        state.mu = family.link.inverse(state.eta + problem.offset)
        state.obj_val = problem.obj_function(state.mu,
                                             start)

        boundary = False
        halved = False  # did we have to halve the step size?

        # if objective function is not finite, keep halving the stepsize until it is finite
        # for the halving step, we probably have to adjust fit$g as well?

        # three checks we'll apply

        def finite_objective(state):
            boundary = True
            halved = True
            return np.isfinite(state.obj_val) and state.obj_val < problem.control.big, boundary, halved

        def valid(state):
            boundary = True
            halved = True
            return valideta(state.eta) and validmu(state.mu), boundary, halved
            
        def decreased_obj(state):
            boundary = False
            halved = True
            return state.obj_val <= state.obj_val_old + 1e-7, boundary, halved

        for test, msg in [(finite_objective,
                           "Non finite objective function! Step size truncated due to divergence."),
                          (valid,
                           "Invalid eta/mu! Step size truncated: out of bounds."),
                          (decreased_obj,
                           "")]:

            if not test(state)[0]:
                if msg:
                    warnings.warn(msg)
                if np.any(np.isnan(coefold)) or np.isnan(intold):
                    raise ValueError("No valid set of coefficients has been found: please supply starting values")

                ii = 1
                check, boundary_, halved_ = test(state)
                boundary = boundary or boundary_
                halved = halved or halved_
                
                while not check:
                    if ii > problem.control.mxitnr:
                        raise ValueError(f"inner loop {test}; cannot correct step size")
                    ii += 1
                    start = (start + coefold)/2
                    start_int = (start_int + intold)/2

                    state.eta = design.get_eta(start,
                                               start_int)

                    state.mu = family.link.inverse(state.eta + problem.offset)
                    state.obj_val = problem.obj_function(state.mu,
                                                         start)
                    check, boundary_, halved_ = test(state)

        # if we did any halving, we have to update the coefficients, intercept
        # and weighted residual in the warm_fit object
        if halved:
            fit.warm_fit.a = start
            fit.warm_fit.aint = start_int
            fit.warm_fit.r =  w * (z - state.eta)

        # test for convergence
        if (np.fabs(state.obj_val - state.obj_val_old)/(0.1 + abs(state.obj_val)) < problem.control.epsnr):
            converged = True
            break

        else:
            coefold = start
            intold = start_int
            state.obj_val_old = state.obj_val

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
        fit.warm_fit = None

    # create a GLMNetResult

    args = asdict(fit)
    args['offset'] = problem.is_offset
    args['nulldev'] = nulldev

    args['dev_ratio'] = (1 - problem.dev_function(state.mu) / nulldev)
    args['family'] = problem.family
    args['converged'] = converged
    args['boundary'] = boundary
    args['obj_function'] = state.obj_val

    return GLMNetResult(**args), problem

