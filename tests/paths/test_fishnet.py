import numpy as np
import pandas as pd
import pytest

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

rng = np.random.default_rng(0)
np_cv_rules = default_converter + numpy2ri.converter

from glmnet import FishNet

def get_glmnet_soln(X,
                    Y,
                    standardize=True,
                    fit_intercept=True,
                    weights=None,
                    offset=None):

    with np_cv_rules.context():

        args = ['family="poisson"']

        if weights is not None:
            rpy.r.assign('weights', weights)
            args.append('weights=weights')
            rpy.r('weights = as.numeric(weights)')

        if standardize:
            rpy.r.assign('standardize', True)
        else:
            rpy.r.assign('standardize', False)
        args.append('standardize=standardize')

        if fit_intercept:
            rpy.r.assign('intercept', True)
        else:
            rpy.r.assign('intercept', False)
        args.append('intercept=intercept')

        if offset is not None:
            rpy.r.assign('offset', offset)
            args.append('offset=offset')

        args = ',\n'.join(args)
        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
    library(glmnet)
    y = as.integer(Y)
    G = glmnet(X, y,
               {args})
    C = as.matrix(coef(G))
    '''
        print(cmd)
        rpy.r(cmd)
        C = rpy.r('C')
    return C.T

def sample1(n):
    return rng.uniform(0, 1, size=n)

@pytest.mark.parametrize('offset', [None, np.zeros, lambda n: sample1(n)]) # should match n=100 below
@pytest.mark.parametrize('sample_weight', [None, lambda n: sample1(n)]) # should match n=100 below
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('n', [1000,50])
@pytest.mark.parametrize('p', [10,100])
def test_fishnet(standardize,
                 fit_intercept,
                 n,
                 p,
                 sample_weight,
                 offset):

    X = rng.standard_normal((n, p))
    Y = rng.poisson(20, size=n)

    D = pd.DataFrame({'Y':Y})
    if offset is not None:
        offset = offset(n)
        offset_id = 'offset'
        D['offset'] = offset
    else:
        offset_id = None

    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        D['weight'] = sample_weight
    else:
        weight_id = None
        

    L = FishNet(standardize=standardize,
                fit_intercept=fit_intercept,
                response_id='Y',
                offset_id=offset_id,
                weight_id=weight_id)

    L.fit(X, D)
    C = get_glmnet_soln(X,
                        Y,
                        standardize=standardize,
                        weights=sample_weight,
                        offset=offset,
                        fit_intercept=fit_intercept)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10

