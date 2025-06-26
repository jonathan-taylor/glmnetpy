"""
Test FishNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet import FishNet, GLM, GLMNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from glmnet.glm import GLMControl
from glmnet.glmnet import GLMNetControl

# rpy2 imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector, IntVector
    from rpy2.robjects import numpy2ri
    
    # Import R packages
    stats = importr('stats')
    glmnet = importr('glmnet')
    
    has_rpy2 = True
except ImportError:
    has_rpy2 = False

# Pytest decorators
ifrpy = pytest.mark.skipif(not has_rpy2, reason='requires rpy2')
alpha = pytest.mark.parametrize('alpha', [0, 0.4, 1])
use_offset = pytest.mark.parametrize('use_offset', [True, False])
use_weights = pytest.mark.parametrize('use_weights', [True, False])
alignment = pytest.mark.parametrize('alignment', ['fraction', 'lambda'])


def numpy_to_r_matrix(X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p = 103, 20
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    R = rng.choice(['D', 'C'], size=n) 
    O = rng.standard_normal(n) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    L = LabelEncoder().fit(R)
    Y = rng.poisson(10, size=n)
    D = np.array([Y, O, W]).T
    Df = pd.DataFrame(D, columns=['response', 'offset', 'weight'])
    return X, Y, O, W, R, D, Df, L


@ifrpy
@use_offset
@use_weights
def test_glm_comparison(sample_data, use_offset, use_weights):
    """Test GLM comparison with different offset and weight combinations."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Configure Python GLM
    glm_kwargs = {'family': sm.families.Poisson()}
    if use_offset:
        glm_kwargs['offset_id'] = 'offset'
    if use_weights:
        glm_kwargs['weight_id'] = 'weight'
    
    G = GLM(response_id='response', **glm_kwargs)
    G.fit(X, Df)
    
    # Configure R glm
    Y_int = Y.astype(int)
    W_numeric = W.astype(float)
    
    r_data_dict = {'Y': IntVector(Y_int)}
    r_kwargs = {'family': stats.poisson}
    
    if use_offset:
        r_data_dict['O'] = FloatVector(O)
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        r_data_dict['W'] = FloatVector(W_numeric)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    # Add predictor columns
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, **r_kwargs)
    r_coef = np.array(stats.coef(r_model))
    
    # Compare results
    assert np.allclose(G.coef_, r_coef[1:])
    assert np.allclose(G.intercept_, r_coef[0])


@ifrpy
@alpha
@use_offset
@use_weights
def test_glmnet_comparison(sample_data, alpha, use_offset, use_weights):
    """Test GLMNet comparison with R glmnet."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Configure Python GLMNet
    glmnet_kwargs = {'family': sm.families.Poisson(), 'alpha': alpha}
    if use_offset:
        glmnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        glmnet_kwargs['weight_id'] = 'weight'
    
    GN = GLMNet(response_id='response', **glmnet_kwargs)
    GN.fit(X, Df)
    
    # Configure R glmnet
    W_numeric = W.astype(float)
    
    r_kwargs = {'family': 'poisson', 'alpha': alpha}
    if use_offset:
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), IntVector(Y), **r_kwargs)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    # Compare results (using index 30 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[30][1:], GN.coefs_[30], rtol=1e-4, atol=1e-4)


@ifrpy
@alpha
@use_offset
@use_weights
def test_fishnet_comparison(sample_data, alpha, use_offset, use_weights):
    """Test FishNet comparison with different alpha, offset, and weight combinations."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Configure Python FishNet
    fishnet_kwargs = {'alpha': alpha}
    if use_offset:
        fishnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        fishnet_kwargs['weight_id'] = 'weight'
    
    GN = FishNet(response_id='response', **fishnet_kwargs)
    GN.fit(X, Df)
    
    # Configure R glmnet
    W_numeric = W.astype(float)
    
    r_kwargs = {'family': 'poisson', 'alpha': alpha}
    if use_offset:
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), IntVector(Y), **r_kwargs)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    # Compare results
    assert np.allclose(r_coef.T[:, 1:], GN.coefs_)
    assert np.allclose(r_coef[0], GN.intercepts_)


@ifrpy
@alpha
@alignment
@use_offset
@use_weights
def test_cross_validation(sample_data, alpha, alignment, use_offset, use_weights):
    """Test cross-validation with different alpha, alignment, offset, and weight combinations."""
    X, Y, O, W, R, D, Df, L = sample_data
    
    # Configure Python FishNet with CV
    fishnet_kwargs = {'response_id': 'response', 'alpha': alpha}
    if use_offset:
        fishnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        fishnet_kwargs['weight_id'] = 'weight'
    
    GN = FishNet(**fishnet_kwargs)
    GN.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN.cross_validation_path(X, Df, cv=cv, alignment=alignment)
    
    # Configure R cv.glmnet
    W_numeric = W.astype(float)
    O_numeric = O.astype(float)
    
    r_kwargs = {
        'foldid': IntVector(foldid.astype(int)),
        'family': 'poisson',
        'alignment': alignment,
        'grouped': True,
        'alpha': alpha
    }
    if use_offset:
        r_kwargs['offset'] = FloatVector(O_numeric)
    if use_weights:
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X), IntVector(Y), **r_kwargs)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN.cv_scores_['Poisson Deviance'], r_cvm)
    assert np.allclose(GN.cv_scores_['SD(Poisson Deviance)'], r_cvsd) 