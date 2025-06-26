"""
Test GaussNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet import GaussNet, GLM, GLMNet, GLMFamilySpec
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
stats = importr('stats')
glmnet = importr('glmnet')


def numpy_to_r_matrix(X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p = 103, 20
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    O = rng.standard_normal(n) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    Y = rng.standard_normal(n) * 5
    D = np.array([Y, O, W]).T
    Df = pd.DataFrame(D, columns=['response', 'offset', 'weight'])
    return X, Y, O, W, D, Df


def test_glm_comparison_with_offset_weight(sample_data):
    """Test GLM comparison with offset and weights."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GLM
    G2 = GLM(response_id='response', offset_id='offset', weight_id='weight',
              family=GLMFamilySpec(), summarize=True)
    G2.fit(X, Df)
    
    # R glm - create data frame with variables
    W_numeric = W.astype(float)
    # Create individual columns for each predictor
    r_data_dict = {'Y': FloatVector(Y), 'O': FloatVector(O), 'W': FloatVector(W_numeric)}
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, weights=FloatVector(W_numeric), 
                       offset=FloatVector(O))
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G2.coef_, r_coef[1:])
    assert np.allclose(G2.intercept_, r_coef[0])

def test_glm_comparison_no_weights(sample_data):
    """Test GLM comparison without weights."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GLM
    G4 = GLM(response_id='response', family=GLMFamilySpec())
    G4.fit(X, Df)
    
    # R glm - create data frame with variables
    # Create individual columns for each predictor
    r_data_dict = {'Y': FloatVector(Y)}
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, family=stats.gaussian)
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G4.coef_, r_coef[1:], rtol=1e-5, atol=1e-5)
    assert np.allclose(G4.intercept_, r_coef[0], rtol=1e-5, atol=1e-5)


def test_glm_comparison_weights_only(sample_data):
    """Test GLM comparison with weights only."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GLM
    G5 = GLM(response_id='response', weight_id='weight', family=GLMFamilySpec())
    G5.fit(X, Df)
    
    # R glm - create data frame with variables
    W_numeric = W.astype(float)
    # Create individual columns for each predictor
    r_data_dict = {'Y': FloatVector(Y), 'W': FloatVector(W_numeric)}
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, weights=FloatVector(W_numeric), 
                       family=stats.gaussian)
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G5.coef_, r_coef[1:])
    assert np.allclose(G5.intercept_, r_coef[0])


def test_glm_comparison_offset_only(sample_data):
    """Test GLM comparison with offset only."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GLM
    G6 = GLM(response_id='response', offset_id='offset', family=GLMFamilySpec())
    G6.fit(X, Df)
    
    # R glm - create data frame with variables
    # Create individual columns for each predictor
    r_data_dict = {'Y': FloatVector(Y), 'O': FloatVector(O)}
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, offset=FloatVector(O), 
                       family=stats.gaussian)
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G6.coef_, r_coef[1:])
    assert np.allclose(G6.intercept_, r_coef[0])


def test_glmnet_comparison(sample_data):
    """Test GLMNet comparison with R glmnet."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GLMNet
    GN = GLMNet(response_id='response', offset_id='offset', weight_id='weight',
                family=GLMFamilySpec())
    GN.fit(X, Df)
    
    # R glmnet
    W_numeric = W.astype(float)
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), 
                        FloatVector(Y), offset=FloatVector(O), 
                        weights=FloatVector(W_numeric), family='gaussian')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    # Compare results (using index 30 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[30][1:], GN.coefs_[30], rtol=1e-3, atol=1e-3)


def test_gaussnet_comparison_with_offset_weight(sample_data):
    """Test GaussNet comparison with offset and weights."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet
    GN2 = GaussNet(response_id='response', offset_id='offset', weight_id='weight')
    GN2.fit(X, Df)
    
    # R glmnet
    W_numeric = W.astype(float)
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          FloatVector(Y), weights=FloatVector(W_numeric), 
                          offset=FloatVector(O), family='gaussian')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn2)))
    
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[:, 1:], GN2.coefs_)
    assert np.allclose(r_coef[0], GN2.intercepts_)


def test_gaussnet_comparison_weights_only(sample_data):
    """Test GaussNet comparison with weights only."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet
    GN2 = GaussNet(response_id='response', weight_id='weight')
    GN2.fit(X, Df)
    
    # R glmnet
    W_numeric = W.astype(float)
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          FloatVector(Y), weights=FloatVector(W_numeric), 
                          family='gaussian')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn2)))
    
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[:, 1:], GN2.coefs_)
    assert np.allclose(r_coef[0], GN2.intercepts_)


def test_gaussnet_comparison_offset_only(sample_data):
    """Test GaussNet comparison with offset only."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet
    GN2 = GaussNet(response_id='response', offset_id='offset')
    GN2.fit(X, Df)
    
    # R glmnet
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(X), 
                          FloatVector(Y), offset=FloatVector(O), 
                          family='gaussian')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn2)))
    
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[:, 1:], GN2.coefs_)
    assert np.allclose(r_coef[0], GN2.intercepts_)


def test_cross_validation_fraction_alignment(sample_data):
    """Test cross-validation with fraction alignment."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet with CV
    GN3 = GaussNet(response_id='response', offset_id='offset')
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             FloatVector(Y), offset=FloatVector(O), foldid=r_foldid,
                             family='gaussian', alignment='fraction', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.cv_scores_['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_lambda_alignment(sample_data):
    """Test cross-validation with lambda alignment."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet with CV
    GN3 = GaussNet(response_id='response', offset_id='offset')
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             FloatVector(Y), offset=FloatVector(O), foldid=r_foldid,
                             family='gaussian', alignment='lambda', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.cv_scores_['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_fraction(sample_data):
    """Test cross-validation with weights using fraction alignment."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet with CV
    GN4 = GaussNet(response_id='response', offset_id='offset', weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             FloatVector(Y), offset=FloatVector(O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='gaussian', alignment='fraction', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.cv_scores_['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_lambda(sample_data):
    """Test cross-validation with weights using lambda alignment."""
    X, Y, O, W, D, Df = sample_data
    
    # Python GaussNet with CV
    GN4 = GaussNet(response_id='response', offset_id='offset', weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X),
                             FloatVector(Y), offset=FloatVector(O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='gaussian', alignment='lambda', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.cv_scores_['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3) 
