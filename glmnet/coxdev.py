from typing import Optional
import numpy as np
import pandas as pd

from dataclasses import (dataclass,
                         InitVar)

@dataclass
class right_censored(object):

    event_time: InitVar(np.ndarray)
    status: InitVar(np.ndarray)
    sample_weight: Optional[np.ndarray] = None
    strata: Optional[np.ndarray] = None
    
    def __post_init__(self,
                      event_time,
                      status):
        self.data = pd.DataFrame({'event_time': event_time,
                                  'status': status})
        if self.sample_weight is None:
            self.data['sample_weight'] = 1
        if self.strata is None:
            self.data['strata'] = 1

        self.data = self.data.sort_values(by=['event_time', 'status'],
                                          ascending=[True, False])
        self._order = np.asarray(self.data.index)

        first_idx, ties = _fid(self.data['event_time'])
        _event = np.zeros_like(self.data['event_time'])
        _event[first_idx] = 1

        _weight_sum = self.data[['event_time', 'sample_weight']].groupby('event_time').sum()['sample_weight']
        _derived_weight = np.zeros(self.data.shape[0])
        _derived_weight[first_idx] = _weight_sum
        self.data['_derived_weight'] = _derived_weight

        _derived_death = np.zeros(self.data.shape[0], bool)
        _derived_death[first_idx] = 1
        self.data['_derived_death'] = _derived_death
        self.data['_risk_count'] = np.cumsum(self.data['_derived_death'])

    def __call__(self,
                 linear_predictor,
                 diag_hessian=True):

        eta = linear_predictor[self._order] # shorthand
        w = self.data['sample_weight'] # shorthand
        d = self.data['status'] # shorthand
        rskcount = self.data['_risk_count']
        eta -= eta.mean()
        exp_eta = np.exp(eta)

        _derived_w = self.data['_derived_weight']
        _derived_d = self.data['_derived_death']
        rskden = np.cumsum((exp_eta*w)[::-1])[::-1]
        rskdeninv = np.hstack([0, np.cumsum((_derived_w / rskden)[_derived_d==1])])

        grad = w * (d - exp_eta * rskdeninv[rskcount])
        return grad
    
def _fid(times):

    '''
    >>> fid([1,4,5,6])
    (array([0, 1, 2, 3]), {})
    >>> fid([1, 1, 1, 2, 3, 3, 4, 4, 4])
    (array([0, 3, 4, 6]), {1: array([0, 1, 2]), 3: array([4, 5]), 4: array([6, 7, 8])})
    >>>
    '''

    times = np.asarray(times)
    unique_times, idx, counts = np.unique(times,
                                          return_index=True,
                                          return_counts=True)
    indices = {}
    for t, c in zip(unique_times,
                    counts):
        if c > 1:
            indices[t] = np.nonzero(times == t)[0]
    return idx, indices

if __name__ == "__main__":

    time = np.array([2,4,3,5,6.5,7,7,7,7,])
    status = np.array([0,1,1,0,0,1,7,7,7])

    rc = right_censored(time, status)
    rc(np.ones_like(time))
    
