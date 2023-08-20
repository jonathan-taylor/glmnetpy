import numpy as np
from copy import copy
import scipy.sparse
import pickle
from glmnet.elnet_fit import elnet_fit, ElNetSpec

from importlib.resources import files

def test_elnet(num=85):

    fname = f"test_{num:03d}.pickle"
    fb = files('glmnet').joinpath('test_data', fname).read_bytes()
    D = pickle.loads(fb)

    args = copy(D['elnet_args'])

    # fix up names of args

    renames = [('x', 'X'),
               ('penalty.factor', 'penalty_factor'),
               ('upper.limits', 'upper_limits'),
               ('lower.limits', 'lower_limits'),
               ('lambda', 'lambda_val'),
               ('save.fit', 'save_fit'),
               ]

    for old, new in renames:
        args[new] = args[old]
        del(args[old])

    args['weights'] = np.asarray(args['weights'])
    thresh = args['thresh']; del(args['thresh'])
    save_fit = args['save_fit']; del(args['save_fit'])

    # make it a sparse array instead of matrix
    
    if scipy.sparse.issparse(args['X']):
        args['X'] = scipy.sparse.csc_array(args['X'])

    spec = ElNetSpec(**args)

    spec_wls_args = spec.wls_args()[0]
    wls_args = D['wls_args']

    for k in spec_wls_args.keys():
        print(k, wls_args[k], spec_wls_args[k])
