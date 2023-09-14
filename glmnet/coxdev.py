from typing import Optional
import numpy as np
import pandas as pd

from dataclasses import (dataclass,
                         InitVar)

@dataclass
class CoxDevianceResult(object):

    deviance: float
    grad: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]

@dataclass
class CoxRightCensored(object):

    event_time: InitVar(np.ndarray)
    status: InitVar(np.ndarray)
    start_time: InitVar(np.ndarray) = None
    sample_weight: Optional[np.ndarray] = None
    diag_hessian: bool = False
    strata: Optional[np.ndarray] = None
    
    def __post_init__(self,
                      event_time,
                      status,
                      start_time):
        self.data = pd.DataFrame({'event_time': event_time,
                                  'status': status})

        if self.sample_weight is None:
            self.data['sample_weight'] = 1
        if self.strata is None:
            self.data['strata'] = 1

        if start_time is not None:
            self.data['start_time'] = start_time # implicitly this is -np.inf if not supplied

        self.data = self.data.sort_values(by=['event_time', 'status'],
                                          ascending=[True, False])
        self._event_order = np.asarray(self.data.index)
        self.data = self.data.reset_index()

        _events = self.data['status'] == 1
        first_idx, ties = _fid(self.data.loc[_events, 'event_time'], np.arange(self.data.shape[0])[_events])

        _weight_sum = self.data.loc[_events, ['event_time', 'sample_weight']].groupby('event_time').sum()['sample_weight']
        _derived_weight = self.data['sample_weight'].copy()
        _derived_weight[first_idx] = _weight_sum
        self.data['_derived_weight'] = _derived_weight

        # compute saturated loglikelihood

        _weight_sum = _weight_sum[_weight_sum>0]
        self._loglik_sat = - np.sum(_weight_sum * np.log(_weight_sum))

        _derived_events = self.data['status'].copy()
        for v in ties.values():
            _derived_events[v] = 0
        _derived_events[first_idx] = 1
        self.data['_derived_events'] = _derived_events
        # this counts how many
        # "1" events occur __at or before__ the current event time
        # becomes a pointer into an array of (reversed) cumulative weights at each
        # "1" event
        self.data['_event_risk_count'] = np.cumsum(self.data['_derived_events']) 
        
        if start_time is not None:
            # note that this is how the "event" sorted "start_times" are ordered
            # NOT how the raw "start_times" are ordered!!
            
            self._start_order = np.sort(self.data['start_time']) 
        else:
            self._start_order = None
            
    def __call__(self,
                 linear_predictor,
                 compute_gradient=True,
                 compute_diag_hessian=True):

        eta_event = linear_predictor[self._event_order] # shorthand
        eta_event -= eta_event.mean()

        w = self.data['sample_weight'] # shorthand
        d = self.data['status'] # shorthand
        event_count = self.data['_event_risk_count']

        exp_eta_event = np.exp(eta_event)
        exp_eta_w = exp_eta_event * w
        _derived_w = self.data['_derived_weight']
        _derived_d = self.data['_derived_events']
        event_den = np.cumsum(exp_eta_w[::-1])[::-1] # sum of exp(eta)*w over events after or at i-th event

        if self._start_order is not None:
            # recall that _start_order orders the "start_times" AFTER HAVING ORDERED by "stop_time"
            exp_eta_w_start = exp_eta_w[self._start_order]
            start_den_tmp = np.cumsum(exp_eta_w_start[::-1])[::-1]
            start_den[self._start_order] = start_den_tmp
            event_den -= start_den

        log_terms = (_derived_w * np.log(event_den))[_derived_d > 0]
        loglik = np.sum((w * eta_event)[d > 0]) - np.sum(log_terms)

        if compute_gradient or compute_diag_hessian:
            event_recip = np.hstack([0, np.cumsum((_derived_w / event_den)[_derived_d==1])])
            if compute_gradient:
                grad = np.array(w * (d - exp_eta_event * event_recip[event_count]))
                grad_cp = np.empty_like(grad)
                grad_cp[self._event_order] = grad
                # 2 for deviance, though gets cancelled by diag_hessian...
                grad = 2 * grad_cp
            else:
                grad = None

            if compute_diag_hessian:
                event_recip_sq = np.cumsum((_derived_w/(event_den**2))[_derived_d==1])
                event_recip_sq = np.hstack([0, event_recip_sq])
                diag_hessian = exp_eta_w**2 * event_recip_sq[event_count] - exp_eta_w * event_recip[event_count]
                diag_hessian_cp = np.empty_like(diag_hessian)
                diag_hessian_cp[self._event_order] = diag_hessian
                # 2 for deviance, though gets cancelled by diag_hessian...
                diag_hessian = 2 * diag_hessian_cp
            else:
                diag_hessian = None
        else:
            grad = diag_hessian = None
        return CoxDevianceResult(deviance=2*(self._loglik_sat - loglik),
                                 grad=grad, 
                                 diag_hessian=diag_hessian)
    
# @dataclass
# class CoxStartStop(object):

#     start_time: InitVar(np.ndarray)
#     stop_time:  InitVar(np.ndarray)
#     status: InitVar(np.ndarray)
#     sample_weight: Optional[np.ndarray] = None
#     diag_hessian: bool = False
#     strata: Optional[np.ndarray] = None
    
#     def __post_init__(self,
#                       start_time,
#                       stop_time,
#                       status):
#         self.data = pd.DataFrame({'start_time': start_time,
#                                   'stop_time': stop_time,
#                                   'status': status})
#         if self.sample_weight is None:
#             self.data['sample_weight'] = 1
#         if self.strata is None:
#             self.data['strata'] = 1

#         stop_data = self.data.sort_values(by=['start_time', 'status'],
#                                           ascending=[True, False])
#         self._start_order = np.sort(self.data['start_time'], ascending=True)
#         self._stop_order = np.asarray(stop_data.index)
#         self.data = stop_data.reset_index()

#         _events = self.data['status'] == 1
#         first_idx, ties = _fid(self.data.loc[_events, 'event_time'], np.arange(self.data.shape[0])[_events])

#         _weight_sum = self.data.loc[_events, ['event_time', 'sample_weight']].groupby('event_time').sum()['sample_weight']
#         _derived_weight = self.data['sample_weight'].copy()
#         _derived_weight[first_idx] = _weight_sum
#         self.data['_derived_weight'] = _derived_weight

#         # compute saturated loglikelihood

#         _weight_sum = _weight_sum[_weight_sum>0]
#         self._loglik_sat = - np.sum(_weight_sum * np.log(_weight_sum))

#         _derived_events = self.data['status'].copy()
#         for v in ties.values():
#             _derived_events[v] = 0
#         _derived_events[first_idx] = 1
#         self.data['_derived_events'] = _derived_events
        
#     def __call__(self,
#                  linear_predictor,
#                  gradient=True,
#                  hessian=True):

#         eta = linear_predictor[self._order] # shorthand
#         w = self.data['sample_weight'] # shorthand
#         d = self.data['status'] # shorthand
#         rskcount = self.data['_event_risk_count']
#         eta -= eta.mean()
#         exp_eta = np.exp(eta)

#         _derived_w = self.data['_derived_weight']
#         _derived_d = self.data['_derived_events']
#         rskden = np.cumsum((exp_eta*w)[::-1])[::-1]

#         log_terms = (_derived_w * np.log(rskden))[_derived_d > 0]
#         loglik = np.sum((w * eta)[d > 0]) - np.sum(log_terms)

#         if gradient or hessian:
#             rskdeninv = np.hstack([0, np.cumsum((_derived_w / rskden)[_derived_d==1])])
#             if gradient:
#                 grad = np.array(w * (d - exp_eta * rskdeninv[rskcount]))
#                 grad_cp = np.empty_like(grad)
#                 grad_cp[self._order] = grad
#             else:
#                 grad_cp = None

#             if hessian:
#                 rskdeninv2 = np.cumsum((_derived_w/(rskden**2))[_derived_d==1])
#                 rskdeninv2 = np.hstack([0, rskdeninv2])
#                 w_exp_eta = w * exp_eta
#                 diag_hessian = w_exp_eta**2 * rskdeninv2[rskcount] - w_exp_eta * rskdeninv[rskcount]
#                 diag_hessian_cp = np.empty_like(diag_hessian)
#                 diag_hessian_cp[self._order] = diag_hessian
#             else:
#                 diag_hessian = None

#         return CoxDevianceResult(deviance=2*(self._loglik_sat - loglik),
#                                  grad=2*grad_cp,  # 2 for deviance, though gets cancelled by diag_hessian...
#                                  diag_hessian=diag_hessian_cp*2)
    
def _fid(times, index=None):

    '''
    >>> fid([1,4,5,6])
    (array([0, 1, 2, 3]), {})
    >>> fid([1, 1, 1, 2, 3, 3, 4, 4, 4])
    (array([0, 3, 4, 6]), {1: array([0, 1, 2]), 3: array([4, 5]), 4: array([6, 7, 8])})
    >>>
    '''

    if index is None:
        index = np.arange(times.shape[0])

    times = np.asarray(times)
    unique_times, first_idx, counts = np.unique(times,
                                                return_index=True,
                                                return_counts=True)
    indices = {}
    for t, c in zip(unique_times,
                    counts):
        if c > 1:
            indices[t] = index[np.nonzero(times == t)[0]]
    return index[first_idx], indices

if __name__ == "__main__":

    time = np.array([2,4,3,5,6.5,7,7,7,7,])
    status = np.array([0,1,1,0,0,1,7,7,7])

    rc = right_censored(time, status)
    rc(np.ones_like(time))
    
