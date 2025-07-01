import warnings
import numpy as np
import pandas as pd

from sklearn.utils import check_X_y

from dataclasses import fields

def _get_data(estimator,
              X,
              y,
              offset_id=None,
              weight_id=None,
              response_id=None,
              check=True,
              multi_output=True # always true below
              ):

    weight = None
    if offset_id is None and weight_id is None:
        # col could be 0 so check for None
        if response_id is not None:
            if isinstance(y, pd.DataFrame):
                response = y.loc[:,response_id]
            else:
                response = y[:,response_id]
        else:
            response = y
        if check:
            X, _ = check_X_y(X, response,
                             accept_sparse=['csc'],
                             multi_output=True,
                             estimator=estimator)
        offset, weight = None, None
    else:
        if isinstance(y, pd.DataFrame):
            response = y
            if offset_id is not None:
                offset = np.asarray(y.loc[:,offset_id])
                if type(offset_id) in [str, int]:
                    response = response.drop(columns=[offset_id])
                else:
                    response = response.drop(columns=offset_id)
            else:
                offset = None
            if weight_id is not None:
                weight = np.asarray(y.loc[:,weight_id])
                if type(weight_id) in [str, int]:
                    response = response.drop(columns=[weight_id])
                else:
                    response = response.drop(columns=weight_id)
            else:
                weight = None
        else:
            keep = np.ones(y.shape[1], bool)
            if offset_id is not None:
                offset = y[:,offset_id]
                keep[offset_id] = 0
            else:
                offset = None
            if weight_id is not None:
                weight = y[:,weight_id]
                keep[weight_id] = 0
            else:
                weight = None

        if check:
            X, _ = check_X_y(X, y,
                             accept_sparse=['csc'],
                             multi_output=True,
                             estimator=estimator)

    # col could be 0 so check for None
    if response_id is not None:
        if isinstance(y, pd.DataFrame):
            response = y.loc[:,response_id]
        else:
            response = y[:,response_id]
    else:
        # we already removed columns of the data frame
        if not isinstance(y, pd.DataFrame):
            if offset_id is not None or weight_id is not None: 
                response = y[:,keep]
            else:
                response = np.asarray(y)
    if weight is None:
        weight = np.ones(y.shape[0])
    return X, y, np.squeeze(np.asarray(response)), offset, weight

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

def _parent_dataclass_from_child(cls,
                                 parent_dict,
                                 **modified_args):
    _fields = [f.name for f in fields(cls)]
    _cls_args = {k:parent_dict[k] for k in parent_dict.keys() if k in _fields}
    _cls_args.update(**modified_args)
    return cls(**_cls_args)

# ===================================================================
# ARGUMENT DICTIONARY EXTRACTED FROM C++ WRAPPER FILES
# ===================================================================

# Using the dictionary structure you suggested for better organization
ALL_CPP_ARGS = {
    # From: elnet_point.cpp
    "elnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],
    "spelnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x_data_array',
        'x_indices_array', 'x_indptr_array', 'xm', 'xs', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],

    # From: lognet.cpp
    "lognet": [
        'parm', 'ni', 'no', 'x', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "splognet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: gaussnet.cpp
    "gaussnet": [
        'ka', 'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spgaussnet": [
        'ka', 'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'pb', 'lmu', 'a0', 'ca',
        'ia', 'nin', 'rsq', 'maxit', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    
    # From: multigaussnet.cpp
    "multigaussnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spmultigaussnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: fishnet.cpp
    "fishnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spfishnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ]
}


# ===================================================================
# VALIDATION FUNCTION (Unchanged, it works with the new structure)
# ===================================================================

def _validate_cpp_args(
    args_dict: dict,
    function_name: str
) -> bool:
    """
    Compares a dictionary of arguments against a list of expected names.
    """
    expected_set = set(ALL_CPP_ARGS[function_name])
    actual_set = set(args_dict.keys())

    missing_args = expected_set - actual_set
    extra_args = actual_set - expected_set

    msg = ''
    if missing_args:
        msg = f"ERROR: The following arguments for {function_name} are MISSING: {sorted(list(missing_args))}, "
    if extra_args:
        msg += f"ERROR: The following EXTRA arguments for {function_name} were found: {sorted(list(extra_args))}"
    if msg:
        return msg

