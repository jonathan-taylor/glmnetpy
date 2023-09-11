from dataclasses import fields, dataclass

_docstrings = {
    'X':'''
X: Union[np.ndarray, scipy.sparse, DesignSpec]
    Input matrix, of shape `(nobs, nvars)`; each row is an observation
    vector. If it is a sparse matrix, it is assumed to be
    unstandardized.  If it is not a sparse matrix, a copy is made and
    standardized.''',

    'y':'''
y: np.ndarray
    Response variable.''',

    'dtype':'''
dtype: np.dtype
    The dtype for Design as a LinearOperator.
    ''',
    
    'score_method':'''
score_method: str
    Which score to use?
    ''',

    'score_method_gaussian':'''
score_method: str
    Which score to use? One of ["r2", "deviance"].
    ''',

    'score_method_binary':'''
score_method: str
    Which score to use? One of ["r2", "accuracy"].
    ''',

    'logging':'''
logging: bool
    Write info and debug messages to log?''',
    
    'standardize':'''
standardize: bool
    Standardize columns of X according to weights? Default is False.''',
    
    'weights':'''
weights: Optional[np.ndarray]
    Observation weights. These are not standardized in the fit.''',

    'sample_weight':'''
sample_weight: Optional[np.ndarray]
    Sample weights.''',
    
    'lambda_val':'''
lambda_val: float
    A single value for the `lambda` hyperparameter.''',

    'lambda_values':'''
lambda_values: np.ndarray
    An array of `lambda` hyperparameters.''',

    'lambda_fractional':'''
lambda_fractional: bool
    Are `lambda_values` a fraction of `lambda_max` or absolute?.''',

    'alpha':r'''
alpha: Optional[float]
    The elasticnet mixing parameter in [0,1].  The penalty is
    defined as $(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.$
    `alpha=1` is the lasso penalty, and `alpha=0` the ridge
    penalty. Defaults to 1.''',

    'fit_intercept':'''
fit_intercept: bool
    Should intercept be fitted (default=`True`) or set to zero (`False`)?''',

    'thresh':'''
thresh: float
    Convergence threshold for coordinate descent. Each inner
    coordinate-descent loop continues until the maximum change in the
    objective after any coefficient update is less than thresh times
    the null deviance.  Default value is `1e-7`.''',

    'maxit':'''
maxit: int
    Maximum number of passes over the data; default is
    `10^5`.  (If a warm start object is provided, the number
    of passes the warm start object performed is included.)
    Default: 25.''',

    'penalty_factor':'''
penalty_factor: np.ndarray (optional)
    Separate penalty factors can be applied to each
    coefficient. This is a number that multiplies `lambda_val` to
    allow differential shrinkage. Can be 0 for some variables,
    which implies no shrinkage, and that variable is always
    included in the model. Default is 1 for all variables (and
    implicitly infinity for variables listed in `exclude`). Note:
    the penalty factors are internally rescaled to sum to
    `nvars=X.shape[1]`.''',

    'exclude':'''
exclude: list
    Indices of variables to be excluded from the model. Default is
    `[]`. Equivalent to an infinite penalty factor.''',

    'lower_limits':'''
lower_limits: Union[float, np.ndarray]
    Vector of lower limits for each coefficient; default
    `-np.inf`. Each of these must be non-positive. Can be
    presented as a single value (which will then be replicated),
    else a vector of length `nvars`.''',

    'upper_limits':'''
upper_limits: Union[float, np.ndarray]
    Vector of upper limits for each coefficient; default
    `np.inf`. See `lower_limits`.''',

    'warm':'''
warm: Optional[Union[ElNetWarmStart,dict]]
    A dict-like with keys `beta` and `a0` containing coefficients
    and intercept respectively which can be used as a warm start.
    For internal use only.''',

    'warm_glm':'''
warm_state: Optional[dict]
    A dict-like with keys `coef_` and `intercept_` containing coefficients
    and intercept respectively which can be used as a warm start.
    For internal use only.''',

    'big':'''
big: float = 9.9e35
    A large float, effectively `np.inf`.''',
    
    'a0':'''
a0: float
    Intercept value.''',
    
    'beta':'''
beta: scipy.sparse.csc_array
    Matrix of coefficients, stored in sparse matrix format.''',
    
    'df':'''
df: int
    The number of nonzero coefficients.''',

    'dim': '''
dim: (int, int)
    Dimension of coefficient matrix.''',

    'dev_ratio': '''
dev_ratio: float
    The fraction of (null) deviance explained.  The deviance
    calculations incorporate weights if present in the model. The
    deviance is defined to be 2*(loglike_sat - loglike), where
    loglike_sat is the log-likelihood for the saturated model (a model
    with a free parameter per observation).  Hence
    dev_ratio=1-dev/nulldev.''',

    'nulldev':'''
nulldev: float
    Null deviance (per observation).  This is defined to be
    2*(loglike_sat -loglike(Null)). The null model refers to the
    intercept model.''',
    
    'npasses':'''
npasses: int
    Total passes over the data.''',

    'jerr':'''
jerr: int
    Error flag, for warnings and errors (largely for internal debugging).''',

    'nobs':'''
nobs: int
    Number of observations.''',
    
    'warm_fit': '''
warm_fit: Optional(ElNetWarmStart)
    Used for warm starts.''',
    
    'warm_glm': '''
warm: Optional(dict, ElNetWarmStart)
    Either a dict-like object with keys "coef_" and "intercept_" or an
    `ElNetWarmStart` instance.''',

    'control_elnet': '''
control: Optional(ElNetControl)
    Parameters to control the solver.''',
    
    'control_glm': '''
control: Optional(GLMControl)

    Parameters to control the solver.
    ''',
    
    'control_glmnet': '''
control: Optional(GLMNetControl)

    Parameters to control the solver.
    ''',

    'control_glmnet_path': '''
control: Optional(GLMNetPathControl)

    Parameters to control the solver.
    ''',

    'mxitnr':'''
mxitnr: int
    Maximum number of quasi Newton iterations.''',

    'epsnr':'''
epsnr: float
    Tolerance for quasi Newton iterations.''',

    'offset':'''
offset: np.ndarray
    Offset for linear predictor.''',
    
    'family':'''
family: Family
    One-parameter exponential family (from `statsmodels`).''',

    'converged':'''
converged: bool
    Did the algorithm converge?''',

    'boundary':'''
boundary: bool
    Was backtracking required due to getting near boundary of valid mean / natural parameters.''',

    'obj_function':'''
obj_function: float
    Value of objective function (deviance + penalty).''',

    'warm_start':'''
warm_start: bool
    When set to True, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just erase the
    previous solution.''',

    'prediction':'''
prediction: np.ndarray
    Predictions on the mean scale for family of a GLM.''',

    'prediction_type':'''
prediction_type: str
    One of "response" or "link". If "response" return a prediction on the mean scale,
    "link" on the link scale. Defaults to "response".''',

    'prediction_type_binomial':'''
prediction_type: str
    One of "response", "link" or "class". If "response" return a prediction on the mean scale,
    "link" on the link scale, and "class" as a class label. Defaults to "class".''',

    'dispersion':'''
dispersion: float
    Dispersion parameter of GLM. If family is Gaussian, will be estimated as 
    minimized deviance divided by degrees of freedom.''',

    'summarize':'''
summarize: bool
    Compute a Wald-type statistical summary from fitted GLM.''',

    # Common attributes

    'coef_':'''
coef_: array-like of shape (n_features,)
    Fitted coefficients.''',

    'intercept_':'''
intercept_: float
    Fitted intercept.''',

    'summary_':'''
summary_: pd.DataFrame
    Results of Wald tests for each fitted coefficient if `summarize` is True, else None.''',

    'covariance_':'''
covariance_: array, shape (n_features, n_features)
    Estimated covariance matrix of fitted intercept and coefficients if `summarize` is True, else None.''',

    'null_deviance_':'''
null_deviance_: float
    Null deviance.''',
    
    'deviance_':'''
deviance_: float
    Deviance of fitted model.''',
    
    'dispersion_':'''
dispersion_: float
    Estimated dispersion of model. For Gaussian this uses the usual unbiased estimate of variance.''',
    
    'regularizer_':'''
regularizer_: object
    Regularizer used in fitting the model. Allows for inspection of parameters of regularizer. For a GLM this is just the 0 function.''',
}


def make_docstring(*fieldnames):

    field_str = '\n\n'.join([_docstrings[f].strip() for f in fieldnames])
    return f'''
Parameters
----------

{field_str}
'''

def add_dataclass_docstring(kls, subs={}):
    """
    Add a docstring to a dataclass using entries in `._docstrings` based on the fields
    of the dataclass.
    """

    fieldnames = [f.name for f in fields(kls)]
    for k in subs:
        fieldnames[fieldnames.index(k)] = subs[k]

    kls.__doc__ = '\n'.join([kls.__doc__, make_docstring(*fieldnames)])
    return kls
