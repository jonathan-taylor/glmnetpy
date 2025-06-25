"""
Test MultiGaussNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet import MultiGaussNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects import numpy2ri

# Import R packages
glmnet = importr('glmnet')


def numpy_to_r_matrix(X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p, q, nlambda = 103, 17, 3, 100
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    O = rng.standard_normal((n, q)) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    Y = rng.standard_normal((n, q)) * 5
    D = np.column_stack([Y, O, W])
    response_id = [f'response[{i}]' for i in range(q)]
    offset_id = [f'offset[{i}]' for i in range(q)]
    Df = pd.DataFrame(D, columns=response_id + offset_id + ['weight'])
    return X, Y, O, W, Df, response_id, offset_id, nlambda


def test_multigaussnet_comparison_with_offset(sample_data):
    """Test MultiGaussNet comparison with offset."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet
    GN2 = MultiGaussNet(response_id=response_id, offset_id=offset_id, nlambda=nlambda)
    GN2.fit(X, Df)
    
    # R glmnet
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          numpy_to_r_matrix(Y), 
                          offset=numpy_to_r_matrix(O),
                          family='mgaussian', nlambda=nlambda)
    r_coef = ro.r.coef(r_gn2)
    
    # Extract coefficients for each response
    C1 = np.array(ro.r['as.matrix'](r_coef.rx2('y1')))
    C2 = np.array(ro.r['as.matrix'](r_coef.rx2('y2')))
    C3 = np.array(ro.r['as.matrix'](r_coef.rx2('y3')))
    
    C = np.array([C1, C2, C3]).T
    
    assert np.allclose(C[:, 1:], GN2.coefs_)
    assert np.allclose(C[:, 0], GN2.intercepts_)


def test_multigaussnet_comparison_weights_only(sample_data):
    """Test MultiGaussNet comparison with weights only."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet
    GN2 = MultiGaussNet(response_id=response_id, weight_id='weight', nlambda=nlambda)
    GN2.fit(X, Df)
    
    # R glmnet
    W_numeric = W.astype(float)
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          numpy_to_r_matrix(Y), 
                          weights=FloatVector(W_numeric), family='mgaussian', nlambda=nlambda)
    r_coef = ro.r.coef(r_gn2)
    
    # Extract coefficients for each response
    C1 = np.array(ro.r['as.matrix'](r_coef.rx2('y1')))
    C2 = np.array(ro.r['as.matrix'](r_coef.rx2('y2')))
    C3 = np.array(ro.r['as.matrix'](r_coef.rx2('y3')))
    
    C = np.array([C1, C2, C3]).T
    
    assert np.allclose(C[:, 1:], GN2.coefs_)
    assert np.allclose(C[:, 0], GN2.intercepts_)


def test_multigaussnet_comparison_with_offset_weight(sample_data):
    """Test MultiGaussNet comparison with offset and weights."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet
    GN2 = MultiGaussNet(response_id=response_id, offset_id=offset_id, weight_id='weight')
    GN2.fit(X, Df)
    
    # R glmnet
    W_numeric = W.astype(float)
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          numpy_to_r_matrix(Y), 
                          weights=FloatVector(W_numeric), 
                          offset=numpy_to_r_matrix(O),
                          family='mgaussian', nlambda=nlambda)
    r_coef = ro.r.coef(r_gn2)
    
    # Extract coefficients for each response
    C1 = np.array(ro.r['as.matrix'](r_coef.rx2('y1')))
    C2 = np.array(ro.r['as.matrix'](r_coef.rx2('y2')))
    C3 = np.array(ro.r['as.matrix'](r_coef.rx2('y3')))
    
    C = np.array([C1, C2, C3]).T
    
    assert np.allclose(C[:, 1:], GN2.coefs_)
    assert np.allclose(C[:, 0], GN2.intercepts_)


def test_cross_validation_fraction_alignment(sample_data):
    """Test cross-validation with fraction alignment."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet with CV
    GN3 = MultiGaussNet(response_id=response_id, offset_id=offset_id)
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=5, alignment='fraction')
    
    # R cv.glmnet
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             numpy_to_r_matrix(Y), offset=numpy_to_r_matrix(O), foldid=r_foldid,
                             family='mgaussian', alignment='fraction', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN3.cv_scores_['Mean Squared Error'], r_cvm)
    assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'], r_cvsd)


def test_cross_validation_lambda_alignment(sample_data):
    """Test cross-validation with lambda alignment."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet with CV
    GN3 = MultiGaussNet(response_id=response_id, offset_id=offset_id)
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=5, alignment='lambda')
    
    # R cv.glmnet
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             numpy_to_r_matrix(Y), offset=numpy_to_r_matrix(O), foldid=r_foldid,
                             family='mgaussian', alignment='lambda', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN3.cv_scores_['Mean Squared Error'], r_cvm)
    assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'], r_cvsd)


def test_cross_validation_with_weights_fraction(sample_data):
    """Test cross-validation with weights using fraction alignment."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet with CV
    GN4 = MultiGaussNet(response_id=response_id, offset_id=offset_id, weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=5, alignment='fraction')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             numpy_to_r_matrix(Y), offset=numpy_to_r_matrix(O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='mgaussian', alignment='fraction', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN4.cv_scores_['Mean Squared Error'], r_cvm)
    assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'], r_cvsd)


def test_cross_validation_with_weights_lambda(sample_data):
    """Test cross-validation with weights using lambda alignment."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiGaussNet with CV
    GN4 = MultiGaussNet(response_id=response_id, offset_id=offset_id, weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=5, alignment='lambda')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             numpy_to_r_matrix(Y), offset=numpy_to_r_matrix(O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='mgaussian', alignment='lambda', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN4.cv_scores_['Mean Squared Error'], r_cvm)
    assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'], r_cvsd) 