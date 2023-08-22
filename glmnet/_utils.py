import numpy as np
import scipy.sparse
import statsmodels.api as sm

from dataclasses import fields

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

def _dataclass_from_parent(cls,
                           parent_dict):
    _fields = [f.name for f in fields(cls)]
    return cls(**{k:parent_dict[k] for k in parent_dict.keys() if k in _fields})

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
    coefficients = coefficients.reshape(-1)
    vp = vp.reshape(-1)
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
    return np.sum(family.resid_dev(y, mu, weights)**2)


