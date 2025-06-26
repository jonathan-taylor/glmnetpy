from dataclasses import (dataclass,
                         asdict,
                         field)
import numpy as np
from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from .docstrings import (add_dataclass_docstring,
                         _docstrings)
from .base import _get_design, DiagonalOperator
from .scoring import (Scorer,
                      mae_scorer,
                      mse_scorer,
                      accuracy_scorer,
                      auc_scorer,
                      aucpr_scorer,
                      ungrouped_mse_scorer,
                      ungrouped_mae_scorer)

@add_dataclass_docstring
@dataclass
class GLMFamilySpec(object):
    
    base: sm_family.Family = field(default_factory=sm_family.Gaussian)

    def __post_init__(self):

        self.is_gaussian = False
        if (isinstance(self.base, sm_family.Gaussian) and
            isinstance(self.base.link, sm_links.Identity)):
            self.is_gaussian = True

        self.is_binomial = False
        if isinstance(self.base, sm_family.Binomial):
            self.is_binomial = True

    def link(self,
             mean_parameter):
        mu = mean_parameter # shorthand
        return self.base.link(mu)
    link.__doc__ = """
Parameters
----------
{mean_parameter}

Returns
-------
{linear_predictor}
""".format(**_docstrings).strip()
    
    def deviance(self,
                 response,
                 mean_parameter,
                 sample_weight=None):
        if sample_weight is not None:
            return self.base.deviance(response,
                                      mean_parameter,
                                      freq_weights=sample_weight)
        else:
            return self.base.deviance(response,
                                      mean_parameter)
    deviance.__doc__ = '''
Parameters
----------

{mean_parameter}
{sample_weight}

Returns
-------
{deviance}
    '''.format(**_docstrings).strip()

    def null_fit(self,
                 response,
                 fit_intercept=True,
                 sample_weight=None,
                 offset=None):

        sample_weight = np.asarray(sample_weight)
        y = np.asarray(response)

        if offset is None:
            offset = np.zeros(y.shape[0])
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        if fit_intercept:

            # solve a one parameter problem

            X1 = np.ones((y.shape[0], 1))
            D = _get_design(X1,
                            sample_weight,
                            standardize=False,
                            intercept=False)
            
            state = GLMState(np.zeros(1),
                             0)
            state.update(D,
                         self,
                         offset,
                         None)

            for i in range(10):

                z, w = self.get_response_and_weights(state,
                                                     y,
                                                     offset,
                                                     sample_weight)
                newcoef = (z*w).sum() / w.sum()
                state = GLMState(np.array([newcoef]),
                                 0)
                state.update(D,
                             self,
                             offset,
                             None)
        else:
            state = GLMState(np.zeros(1), 0)
            state.link_parameter = offset
            state.mean_parameter = self.base.link.inverse(state.link_parameter)
        return state
    null_fit.__doc__ = '''
Parameters
----------

{mean_parameter}
{fit_intercept}
{sample_weight}
{offset}
    
Returns
-------
null_state: GLMState
    Fitted null state of GLM.
    '''.format(**_docstrings).strip()

    def get_null_deviance(self,
                          response,
                          sample_weight=None,
                          offset=None,
                          fit_intercept=True):
        state0 = self.null_fit(response,
                               fit_intercept=fit_intercept,
                               sample_weight=sample_weight,
                               offset=offset)
        D = self.deviance(response,
                          state0.mean_parameter,
                          sample_weight=sample_weight)
        return state0, D

    get_null_deviance.__doc__ = '''
Parameters
----------

{mean_parameter}
{fit_intercept}
{sample_weight}
{offset}
    
Returns
-------
null_state: GLMState
    Fitted null state of GLM.
{deviance}
    '''.format(**_docstrings).strip()

    def get_response_and_weights(self,
                                 state,
                                 response,
                                 offset,
                                 sample_weight):

        y = response # shorthand
        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.link_parameter)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        newton_weights = sample_weight * dmu_deta**2 / varmu

        pseudo_response = state.eta + (y - state.mu) / dmu_deta

        return pseudo_response, newton_weights
        
    get_response_and_weights.__doc__ = '''
Parameters
----------

state: GLMState
    State of GLM.
{response}
{offset}
{sample_weight}
    
Returns
-------
pseudo_response: np.ndarray
    Pseudo-response for (quasi) Newton step.
newton_weights: np.ndarray
    Weights to be used for diagonal in Newton step.
    
    '''.format(**_docstrings).strip()

    def information(self,
                    state,
                    sample_weight=None):

        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.link_parameter)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        W = dmu_deta**2 / varmu
        if sample_weight is not None:
            W *= sample_weight
            
        n = W.shape[0]
        W = W.reshape(-1)
        return DiagonalOperator(W)
    information.__doc__ = '''
Parameters
----------

state: GLMState
    State of GLM.
{sample_weight}
    
Returns
-------
information: DiagonalOperator
    Diagonal information matrix of the response vector for
    `state.mean_pararmeter`.  
    '''.format(**_docstrings).strip()

    # Private methods

    def _default_scorers(self):
        """
        Construct default scorers for GLM.
        """

        fam_name = self.base.__class__.__name__

        def _dev(y, yhat, sample_weight):
            return self.deviance(y, yhat, sample_weight) / y.shape[0]
        dev_scorer = Scorer(name=f'{fam_name} Deviance',
                            score=_dev,
                            maximize=False)
        
        scorers_ = [dev_scorer,
                    mse_scorer,
                    mae_scorer,
                    ungrouped_mse_scorer,
                    ungrouped_mae_scorer]

        if isinstance(self.base, sm_family.Binomial):
            scorers_.extend([accuracy_scorer,
                             auc_scorer,
                             aucpr_scorer])

        return scorers_

    def _get_null_state(self,
                        null_fit,
                        nvars):
        coefold = np.zeros(nvars)   # initial coefs = 0
        state = GLMState(coef=coefold,
                         intercept=null_fit.intercept)
        state.mean_parameter = null_fit.mean_parameter
        state.link_parameter = null_fit.link_parameter
        return state
    
    def link(self,
             responses):
        return self.base.link(responses)

    def predict(self,
                linpred,
                prediction_type='response'):

        if prediction_type == 'link':
            return linpred
        elif prediction_type == 'response':
            return self.base.link.inverse(linpred)
        else:
            raise ValueError("prediction should be one of 'response' or 'link'")


@add_dataclass_docstring
@dataclass
class BinomFamilySpec(GLMFamilySpec):

    base: sm_family.Family = field(default_factory=sm_family.Binomial)

    def predict(self,
                linpred,
                prediction_type='response'):

        if prediction_type == 'link':
            return linpred
        elif prediction_type == 'response':
            return self.base.link.inverse(linpred)
        elif prediction_type == 'class':
            pi_hat = self.base.link.inverse(linpred)
            _integer_classes = (pi_hat > 0.5).astype(int)
            return _integer_classes
        else:
            raise ValueError("prediction should be one of 'response', 'link' or 'class'")


@dataclass
class GLMState(object):

    coef: np.ndarray
    intercept: np.ndarray
    obj_val: float = np.inf
    pmin: float = 1e-9
    
    def __post_init__(self):

        self._stack = np.hstack([self.intercept,
                                 self.coef])

    def update(self,
               design,
               family,
               offset,
               objective=None):
        '''pin the mu/eta values to coef/intercept'''

        family = family.base
        self.linear_predictor = design @ self._stack
        if offset is None:
            self.link_parameter = self.linear_predictor
        else:
            self.link_parameter = self.linear_predictor + offset
        self.mean_parameter = family.link.inverse(self.link_parameter)

        # shorthand
        self.mu = self.mean_parameter 
        self.eta = self.linear_predictor 

        if isinstance(family, sm_family.Binomial):
            self.mu = np.clip(self.mu, self.pmin, 1-self.pmin)
            self.link_parameter = family.link(self.mu)

        if objective is not None:
            self.obj_val = objective(self)
        
    def logl_score(self,
                   family,
                   y,
                   sample_weight):

        family = family.base
        varmu = family.variance(self.mu)
        dmu_deta = family.link.inverse_deriv(self.link_parameter)
        
        # compute working residual
        r = (y - self.mu) 
        return sample_weight * r * dmu_deta / varmu 

