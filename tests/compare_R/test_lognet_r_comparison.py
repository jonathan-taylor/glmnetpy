"""
Test file for comparing LogNet implementation with R glmnet package.
Converts the original Jupyter notebook with R magic cells to proper pytest tests using rpy2.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet.glm import BinomialGLM
from glmnet import LogNet, GLM, GLMNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# rpy2 imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import IntVector, FloatVector
    from rpy2.robjects import Formula
    from rpy2.robjects import DataFrame
    
    # Import R packages
    base = importr('base')
    stats = importr('stats')
    glmnet = importr('glmnet')
    
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    pytest.skip("rpy2 or R glmnet not available", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n, p = 103, 20
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    R = rng.choice(['D', 'C'], size=n) 
    O = rng.standard_normal(n) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    W = W / W.mean()
    L = LabelEncoder().fit(R)
    Y = R == L.classes_[1]
    D = np.array([Y, O, W]).T
    Df = pd.DataFrame(D, columns=['binary', 'offset', 'weight'])
    Df['response'] = R
    
    return X, Y, O, W, R, D, Df, L


def test_glm_comparison_with_offset_weight(sample_data):
    """Test GLM comparison with offset and weights."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python GLM
    G1 = GLM(response_id=0, offset_id=1, weight_id=2, family=sm.families.Binomial())
    G1.fit(X, D)
    
    G2 = BinomialGLM(response_id='response', offset_id='offset', weight_id='weight')
    G2.fit(X, Df)
    
    # Verify Python implementations match
    assert np.allclose(G2.coef_, G1.coef_)
    assert np.allclose(G2.intercept_, G1.intercept_)
    
    # R GLM - create individual columns for X matrix
    notY = 1 - Y.astype(int)
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    # Create R data frame with individual X columns
    r_data = {'Y': IntVector(Y_int), 'O': FloatVector(O), 'W': FloatVector(W_numeric)}
    for i in range(X.shape[1]):
        r_data[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_df = DataFrame(r_data)
    
    # Create formula with all X columns
    x_cols = ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    formula = Formula(f'Y ~ {x_cols}')
    
    r_model = stats.glm(formula, data=r_df, weights=FloatVector(W_numeric), 
                       offset=FloatVector(O), family=stats.binomial())
    r_coef = np.array(stats.coef(r_model))
    
    # Compare Python and R results
    assert np.allclose(G2.coef_, r_coef[1:])
    assert np.allclose(G2.intercept_, r_coef[0])


def test_glm_comparison_no_weights(sample_data):
    """Test GLM comparison without weights."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python GLM
    G4 = BinomialGLM(response_id='binary')
    G4.fit(X, Df)
    
    # R GLM
    Y_int = Y.astype(int)
    notY = (1 - Y).astype(int)
    W_numeric = W.astype(float)
    
    r_data = {'Y': IntVector(Y_int)}
    for i in range(X.shape[1]):
        r_data[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_df = DataFrame(r_data)
    
    x_cols = ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    formula = Formula(f'Y ~ {x_cols}')
    
    r_model = stats.glm(formula, data=r_df, family=stats.binomial())
    r_coef = np.array(stats.coef(r_model))
    
    # Compare results
    assert np.allclose(G4.coef_, r_coef[1:])
    assert np.allclose(G4.intercept_, r_coef[0])


def test_glm_comparison_weights_only(sample_data):
    """Test GLM comparison with weights only."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python GLM
    G5 = BinomialGLM(response_id='binary', weight_id='weight')
    G5.fit(X, Df)
    
    # R GLM
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    r_data = {'Y': IntVector(Y_int), 'W': FloatVector(W_numeric)}
    for i in range(X.shape[1]):
        r_data[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_df = DataFrame(r_data)
    
    x_cols = ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    formula = Formula(f'Y ~ {x_cols}')
    
    r_model = stats.glm(formula, data=r_df, weights=FloatVector(W_numeric), 
                       family=stats.binomial())
    r_coef = np.array(stats.coef(r_model))
    
    # Compare results
    assert np.allclose(G5.coef_, r_coef[1:])
    assert np.allclose(G5.intercept_, r_coef[0])


def test_glm_comparison_offset_only(sample_data):
    """Test GLM comparison with offset only."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python GLM
    G6 = BinomialGLM(response_id='binary', offset_id='offset')
    G6.fit(X, Df)
    
    # R GLM
    notY = (1 - Y).astype(int)
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    r_data = {'Y': IntVector(Y_int), 'O': FloatVector(O)}
    for i in range(X.shape[1]):
        r_data[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_df = DataFrame(r_data)
    
    x_cols = ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    formula = Formula(f'Y ~ {x_cols}')
    
    r_model = stats.glm(formula, data=r_df, offset=FloatVector(O), 
                       family=stats.binomial())
    r_coef = np.array(stats.coef(r_model))
    
    # Compare results
    assert np.allclose(G6.coef_, r_coef[1:])
    assert np.allclose(G6.intercept_, r_coef[0])


def test_glmnet_comparison(sample_data):
    """Test GLMNet comparison with R glmnet."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python GLMNet
    GN = GLMNet(response_id='binary', offset_id='offset', weight_id='weight',
                family=sm.families.Binomial())
    GN.fit(X, Df)
    
    # R glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    r_gn = glmnet.glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                        IntVector(Y_int), offset=FloatVector(O), 
                        weights=FloatVector(W_numeric), family='binomial')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    # Compare results (using index 10 as in original)
    # Handle the case where r_coef might be 1D
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[10][1:], GN.coefs_[10], rtol=1e-4, atol=1e-4)


def test_lognet_comparison(sample_data):
    """Test LogNet comparison with R glmnet."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python LogNet
    GN2 = LogNet(response_id='response', offset_id='offset', weight_id='weight')
    GN2.fit(X, Df)
    
    # R glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    r_gn2 = glmnet.glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                          IntVector(Y_int), weights=FloatVector(W_numeric), 
                          offset=FloatVector(O), family='binomial')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn2)))
    
    # Handle the case where r_coef might be 1D
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    # Compare results
    assert np.allclose(r_coef.T[:, 1:], GN2.coefs_)
    assert np.allclose(r_coef[0], GN2.intercepts_)


def test_cross_validation_fraction_alignment(sample_data):
    """Test cross-validation with fraction alignment."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python LogNet with CV
    GN3 = LogNet(response_id='response', offset_id='offset')
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=5, alignment='fraction')
    
    # R cv.glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    O_numeric = O.astype(float)
    R_numeric = Y.astype(float)
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                             IntVector(Y_int), offset=FloatVector(O_numeric), 
                             foldid=r_foldid, family='binomial', 
                             alignment="fraction", grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN3.cv_scores_['Binomial Deviance'], r_cvm)
    assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'], r_cvsd)


def test_cross_validation_lambda_alignment(sample_data):
    """Test cross-validation with lambda alignment."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python LogNet with CV
    GN3 = LogNet(response_id='response', offset_id='offset')
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=5, alignment='lambda')
    
    # R cv.glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    O_numeric = O.astype(float)
    R_numeric = Y.astype(float)
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                             IntVector(Y_int), offset=FloatVector(O_numeric), 
                             foldid=r_foldid, family='binomial', 
                             alignment="lambda", grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN3.cv_scores_['Binomial Deviance'], r_cvm)
    assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'], r_cvsd)


def test_cross_validation_with_weights_fraction(sample_data):
    """Test cross-validation with weights using fraction alignment."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python LogNet with CV
    GN4 = LogNet(response_id='response', offset_id='offset', weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=5, alignment='fraction')
    
    # R cv.glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    O_numeric = O.astype(float)
    R_numeric = Y.astype(float)
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                             IntVector(Y_int), offset=FloatVector(O_numeric), 
                             weights=FloatVector(W_numeric), foldid=r_foldid, 
                             family='binomial', alignment="fraction", grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN4.cv_scores_['Binomial Deviance'], r_cvm)
    assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'], r_cvsd)


def test_cross_validation_with_weights_lambda(sample_data):
    """Test cross-validation with weights using lambda alignment."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Python LogNet with CV
    GN4 = LogNet(response_id='response', offset_id='offset', weight_id='weight')
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=5, alignment='lambda')
    
    # R cv.glmnet
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    O_numeric = O.astype(float)
    R_numeric = Y.astype(float)
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(ro.r.matrix(FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1]), 
                             IntVector(Y_int), offset=FloatVector(O_numeric), 
                             weights=FloatVector(W_numeric), foldid=r_foldid, 
                             family='binomial', alignment="lambda", grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN4.cv_scores_['Binomial Deviance'], r_cvm)
    assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'], r_cvsd) 