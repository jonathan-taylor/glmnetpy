from typing import Union, List, Optional
from dataclasses import dataclass, field
   
import numpy as np
import scipy.sparse

from .glmnetpp import wls as dense_wls
from .glmnetpp import spwls as sparse_wls

from .design import DesignSpec
from ._utils import _jerr_elnetfit


@dataclass
class ElNetControl(object):

    thresh: float = 1e-7
    maxit: int = 100000
    big: float = 9.9e35
    
@dataclass
class BaseSpec(object):
    
    X: Union[np.ndarray, scipy.sparse._csc.csc_array, DesignSpec]
    y : np.ndarray

@dataclass
class ElNetSpec(BaseSpec):

    lambda_val: float
    alpha: float = 1.0
    lower_limits: Union[float, np.ndarray] = -np.inf
    upper_limits: Union[float, np.ndarray] = np.inf
    exclude: list = field(default_factory=list)
    penalty_factor: Optional[Union[np.ndarray, float]] = None
    weights: Optional[np.ndarray] = None
    intercept: bool = True
    control: ElNetControl = field(default_factory=ElNetControl)

    def __post_init__(self):

        if self.control is None:
            self.control = ElNetControl()
        elif type(self.control) == dict:
            self.control = ElNetControl(**self.control)
        if self.exclude is None:
            self.exclude = []
            
        _set_limits(self)
        _set_vp(self)
        _set_design(self)
        
    def fit(self, warm=None):

        args, nulldev = self._wls_args(warm)

        if scipy.sparse.issparse(self.X):
            wls_fit = sparse_wls(**args)
        else:
            wls_fit = dense_wls(**args)

        nobs, nvars = self.X.shape

        # if error code > 0, fatal error occurred: stop immediately
        # if error code < 0, non-fatal error occurred: return error code

        if wls_fit['jerr'] != 0:
            errmsg = _jerr_elnetfit(wls_fit['jerr'], self.control.maxit)
            raise ValueError(errmsg['msg'])

        warm_args = {}
        for key in ["almc", "r", "xv", "ju", "vp",
                    "cl", "nx", "a", "aint", "g",
                    "ia", "iy", "iz", "mm", "nino",
                    "rsqc", "nlp"]:
                warm_args[key] = wls_fit[key]

        warm_args['m'] = args['m'] # isn't this always 1?
        warm_args['no'] = nobs
        warm_args['ni'] = nvars

        warm_fit = ElNetWarmStart(**warm_args)

        beta = scipy.sparse.csc_array(wls_fit['a']) # shape=(1, nvars)

        out = ElNetResult(a0=wls_fit['aint'],
                          beta=beta,
                          df=np.sum(np.abs(beta) > 0),
                          dim=beta.shape,
                          lambda_val=self.lambda_val,
                          dev_ratio=wls_fit['rsqc'],
                          nulldev=nulldev,
                          npasses=wls_fit['nlp'],
                          jerr=wls_fit['jerr'],
                          nobs=nobs,
                          warm_fit=warm_fit)

        return out

    def _wls_args(self, warm=None):
        return _wls_args(self, warm)


@dataclass
class ElNetWarmStart(object):

    almc: float
    r: np.ndarray
    xv: np.ndarray
    ju: np.ndarray
    vp: np.ndarray
    cl: np.ndarray
    nx: int
    a: np.ndarray
    aint: float
    g: np.ndarray
    ia: np.ndarray
    iy: np.ndarray
    iz: int
    mm: np.ndarray
    nino: int
    rsqc: float
    nlp: int
    m: int
    no: int
    ni: int

@dataclass
class ElNetResult(object):

    '''
    a0: Intercept value

    beta: matrix of coefficients, stored in sparse matrix format

    df: The number of nonzero coefficients.

    dim: Dimension of coefficient matrix.

    lambda_val: Lambda value used.

    dev_ratio: The fraction of (null) deviance explained.

        The deviance calculations incorporate weights if present in the model. The deviance is
        defined to be 2*(loglike_sat - loglike), where loglike_sat is the log-likelihood
        for the saturated model (a model with a free parameter per observation).
        Hence dev_ratio=1-dev/nulldev.

    nulldev: Null deviance (per observation).

        This is defined to be 2*(loglike_sat -loglike(Null)). The null
         model refers to the intercept model.}

    npasses: Total passes over the data.

    jerr: Error flag, for warnings and errors (largely for internal debugging).

    nobs: Number of observations.

    warm_fit: Used for warm starts.

    '''

    a0: float 
    beta: scipy.sparse._csc.csc_array 
    df: int
    dim: tuple
    lambda_val: float
    dev_ratio: float
    nulldev: float
    npasses: int
    jerr: int
    nobs: int
    warm_fit: dict

def elnet_fit(X,
              y,
              weights,
              lambda_val,
              alpha=1.0,
              intercept=True,
              penalty_factor=None, 
              exclude=[],
              lower_limits=-np.inf,
              upper_limits=np.inf,
              thresh=1e-7,
              maxit=100000,
              warm=None,
              save_fit=False,
              from_glmnet_fit=False):

    '''A wrapper around a C++ subroutine which minimizes

    .. math::
    
        1/2 \sum w_i (y_i - X_i^T \beta)^2 + \sum \lambda \gamma_j [(1-\alpha)/2 \beta^2+\alpha|\beta|]

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

    warm: ElNetWarmStart(optional)

        A dict-like with keys `beta` and `a0` containing coefficients
        and intercept respectively which can be used as a warm start.
        For internal use only.

    from_glmnet_fit: bool

        Was `elnet_fit` called from `glmnet_fit`?
        Default is `False`.This has implications for computation of the penalty factors.

    save_fit: bool

        Return the warm start object? Default is `False`.

    Returns
    -------

    result: ElNetResult

    '''

    control = ElNetControl(thresh=thresh,
                           maxit=maxit)
    
    problem = ElNetSpec(X=X,
                        y=y,
                        weights=weights,
                        lambda_val=lambda_val,
                        alpha=alpha,
                        intercept=intercept,
                        penalty_factor=penalty_factor,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        exclude=exclude,
                        control=control)
    
    return problem.fit()


def _elnet_args(design,
                y,
                weights,
                lambda_val,
                vp, 
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
                from_glmnet_fit=False):
    
    X = design.X
        
    exclude = np.asarray(exclude, np.int32)

    nobs, nvars = X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # compute null deviance
    # weights = weights / weights.sum()
    
    ybar = np.sum(y * weights) / np.sum(weights)
    nulldev = np.sum(weights * (y - ybar)**2)

    # if class "glmnetfit" warmstart object provided, pull whatever we want out of it
    # else, prepare arguments, then check if coefs provided as warmstart
    # (if only coefs are given as warmstart, we prepare the other arguments
    # as if no warmstart was provided)

    if isinstance(warm, ElNetWarmStart): # assumes it is a dictionary like `warm_fit`
        a = warm.a
        aint = warm.aint
        alm0 = warm.almc
        cl = warm.cl
        g = warm.g.reshape((-1,1))
        ia = warm.ia
        iy = warm.iy
        iz = warm.iz
        ju = warm.ju.reshape((-1,1))
        m = warm.m
        mm = warm.mm.reshape((-1,1))
        nino = warm.nino
        nobs = warm.no
        nvars = warm.ni
        nlp = warm.nlp
        nx = warm.nx
        r = warm.r.reshape((-1,1))
        rsqc = warm.rsqc
        xv = warm.xv.reshape((-1,1))
        vp = warm.vp.reshape((-1,1))
    else:
        
        # if calling from glmnet.fit(), we do not need to check on exclude
        # and penalty.factor arguments as they have been prepared by glmnet.fit()
        # compute ju
        # assume that there are no constant variables
        ju = np.ones((nvars, 1), np.int32)
        ju[exclude] = 0

        # compute cl from upper and lower limits

        cl = np.asfortranarray([lower_limits,
                                upper_limits], float)

        nx = nvars #  as.integer(nvars)

                                         # From elnet.fit R code
        a  = np.zeros((nvars, 1))        # double(nvars)
        aint = 0.                        # double(1) -- mismatch?
        alm0  = 0.                       # double(1) -- mismatch?
        g = np.zeros((nvars, 1))         # double(nvars) -- mismatch?
        ia = np.zeros((nx, 1), np.int32) # integer(nx)
        iy = np.zeros((nvars, 1), np.int32)   # integer(nvars)     
        iz = 0                           # integer(1) -- mismatch?
        m = 1                            # as.integer(1)
        mm = np.zeros((nvars, 1), np.int32)   # integer(nvars) -- mismatch?
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
                raise ValueError('if not an instance of ElNetWarmStart `warm` should be dict-like with keys "beta" and "a0"')

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

    _args = {'alm0':alm0,
             'almc':almc,
             'alpha':alpha,
             'm':m,
             'no':nobs,
             'ni':nvars,
             'r':r,
             'xv':xv,
             'v':v,
             'intr':intr,
             'ju':ju,
             'vp':vp,
             'cl':cl,
             'nx':nx,
             'thr':thr,
             'maxit':maxit,
             'a':a_new,
             'aint':aint,
             'g':g,
             'ia':ia,
             'iy':iy,
             'iz':iz,
             'mm':mm,
             'nino':nino,
             'rsqc':rsqc,
             'nlp':nlp,
             'jerr':jerr}

    _args.update(**design.wls_args())

    return _args, nulldev

def _set_limits(spec):
    X = spec.X
    nobs, nvars = X.shape

    lower_limits = np.asarray(spec.lower_limits)
    upper_limits = np.asarray(spec.upper_limits)

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
    lower_limits[lower_limits == -np.inf] = -spec.control.big
    upper_limits[upper_limits == np.inf] = spec.control.big

    spec.lower_limits, spec.upper_limits = lower_limits, upper_limits

def _set_vp(spec):

    (penalty_factor,
     exclude) = (spec.penalty_factor,
                 spec.exclude)
    _, nvars = spec.X.shape

    if penalty_factor is None:
        penalty_factor = np.ones(nvars)

    # check and standardize penalty factors (to sum to nvars)
    _isinf_penalty = np.isinf(penalty_factor)

    if np.any(_isinf_penalty):
        exclude.extend(np.nonzero(_isinf_penalty)[0])
        exclude = np.unique(exclude)

    exclude = list(np.asarray(exclude, int))

    if len(exclude) > 0:
        if max(exclude) >= nvars:
            raise ValueError("Some excluded variables out of range")
        penalty_factor[exclude] = 1 # now can change penalty_factor

    vp = np.maximum(0, penalty_factor).reshape((-1,1))
    vp = (vp * nvars / vp.sum())

    spec.exclude, spec.vp = exclude, vp

def _set_design(spec):
    if isinstance(spec.X, DesignSpec):
        design = spec.X
        spec.design = design
        spec.X = spec.design.X
        if spec.weights is None:
            spec.weights = np.ones(spec.X.shape[0])
    else:
        if spec.weights is None:
            spec.weights = np.ones(spec.X.shape[0])
        spec.design = DesignSpec(spec.X, spec.weights)

def _wls_args(spec,
              warm=None):

    return _elnet_args(spec.design,
                       spec.y,
                       spec.weights,
                       spec.lambda_val,
                       spec.vp,
                       alpha=spec.alpha,
                       intercept=spec.intercept,
                       penalty_factor=spec.penalty_factor,
                       exclude=spec.exclude,
                       lower_limits=spec.lower_limits,
                       upper_limits=spec.upper_limits,
                       thresh=spec.control.thresh,
                       maxit=spec.control.maxit,
                       warm=warm)
