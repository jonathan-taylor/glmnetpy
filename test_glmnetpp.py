# Test against R results.
n_tests = 144

import unittest
import glmnet.glmnetpp as gpp
import pickle as pk
import numpy as np
import scipy.sparse as sp

def csc_matrices_equal(mat1, mat2):
    # Check if the structures are the same
    if (mat1.indptr != mat2.indptr).any() or (mat1.indices != mat2.indices).any():
        return False
    # Check if the data is the same
    if (mat1.data != mat2.data).any():
        return False
    return True

def results_same(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            return False
        v1 = dict1[key]
        v2 = dict2[key]
        if isinstance(v1, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        else:
            if isinstance(v1, sp.csc_matrix):
                if not csc_matrices_equal(v1, v2):
                    return False
                else:
                    return True
            else:
                if v1 != v2:
                    return False
                else:
                    return True

class TestAgainstR(unittest.TestCase):

    def test_files(self):
        for i in range(n_tests):
            fname = f"./tests/test_{i+1:03}.pickle"
            with open(fname, "rb") as f:
                test_data = pk.load(f)
                ## test_data is a dict of two elements: args and results (from R elnet.fit call)
                args = test_data['args']
                if test_data['sparse']:
                    print(f"Testing Sparse X case: {fname}")                
                    # Assuming x is a scipy csc_matrix
                    data_array = args['x'].data
                    indices_array = args['x'].indices
                    indptr_array = args['x'].indptr
                    
                    del args['x']  ## remove from dict since we send pointers for sparse mat

                    ## Insert pointer elements into dict
                    args['x_data_array'] = data_array
                    args['x_indices_array'] = indices_array
                    args['x_indptr_array'] = indptr_array
                    
                    result = gpp.spwls(**args)
                else:
                    print(f"Testing Dense X case: {fname}")                
                    result = gpp.wls(**args)
            self.assertTrue(results_same(test_data['result'], result))


if __name__ == '__main__':
    unittest.main()
            
