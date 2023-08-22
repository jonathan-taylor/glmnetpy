_docstrings = {
    'X':'''
    X: Union[np.ndarray, scipy.sparse, DesignSpec]
        Input matrix, of shape `(nobs, nvars)`; each row is an
        observation vector. If it is a sparse matrix, it is assumed to
        be unstandardized.  If it is not a sparse matrix, it is
        assumed that any standardization needed has already been done.
    ''',

    'y':'''
    y: np.ndarray
        Quantitative response variable.
    ''',

    'weights':'''
    weights: np.ndarray
        Observation weights. `elnet_fit` does NOT standardize these weights.
    ''',

    'lambda_val':'''
    lambda_val: float
        A single value for the `lambda` hyperparameter.
    ''',

    'alpha':r'''
    alpha: float

        The elasticnet mixing parameter in [0,1].  The penalty is
        defined as $(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.$
        `alpha=1` is the lasso penalty, and `alpha=0` the ridge
        penalty.
    ''',

    'intercept':'''
    intercept: bool
        Should intercept be fitted (default=`True`) or set to zero (`False`)?
    ''',

    'thresh':'''
    thresh: float

        Convergence threshold for coordinate descent. Each inner
        coordinate-descent loop continues until the maximum change in the
        objective after any coefficient update is less than thresh times
        the null deviance.  Default value is `1e-7`.
    ''',

    'maxit':'''
    maxit: int

        Maximum number of passes over the data; default is
        `10^5`.  (If a warm start object is provided, the number
        of passes the warm start object performed is included.)
    ''',

    'penalty_factor':'''
    penalty_factor: np.ndarray (optional)

        Separate penalty factors can be applied to each
        coefficient. This is a number that multiplies `lambda_val` to
        allow differential shrinkage. Can be 0 for some variables,
        which implies no shrinkage, and that variable is always
        included in the model. Default is 1 for all variables (and
        implicitly infinity for variables listed in `exclude`). Note:
        the penalty factors are internally rescaled to sum to
        `nvars=X.shape[1]`.
    ''',

    'exclude':'''
    exclude: list

        Indices of variables to be excluded from the model. Default is
        `[]`. Equivalent to an infinite penalty factor.
    ''',

    'lower_limits':'''
    lower_limits: Union[float, np.ndarray]

        Vector of lower limits for each coefficient; default
        `-np.inf`. Each of these must be non-positive. Can be
        presented as a single value (which will then be replicated),
        else a vector of length `nvars`.
    ''',

    'upper_limits':'''
    upper_limits: Union[float, np.ndarray]

        Vector of upper limits for each coefficient; default
        `np.inf`. See `lower_limits`.
    ''',

    'warm_el':'''
    warm: Optional[Union[ElNetWarmStart,dict]]

        A dict-like with keys `beta` and `a0` containing coefficients
        and intercept respectively which can be used as a warm start.
        For internal use only.
    ''',

    }

def _make_docstring(*fieldnames):

    return '\n\n'.join([_docstrings[f].strip() for f in fieldnames])
