"""
Test CoxNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet.cox import CoxLM, CoxNet, CoxFamilySpec
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.api as sm
from glmnet.glm import GLMControl
from glmnet.glmnet import GLMNetControl

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects import numpy2ri

# Import R packages
survival = importr('survival')
glmnet = importr('glmnet')


def numpy_to_r_matrix(X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    rng = np.random.default_rng(0)
    n, p = 831, 20
    status = rng.choice([0, 1], size=n)
    start = rng.integers(0, 5, size=n)
    event = start + rng.integers(1, 5, size=n) 
    event_data = pd.DataFrame({'event': event, 'status': status, 'start': start})
    X = rng.standard_normal((n, p))
    W = rng.integers(2, 6, size=n) + rng.uniform(0, 1, size=n)
    W[:20] = 3
    event_data['weight'] = W
    
    # Create Cox family specs
    breslow = CoxFamilySpec(event_data, event_id='event', status_id='status', 
                           start_id='start', tie_breaking='breslow')
    efron = CoxFamilySpec(event_data, event_id='event', status_id='status', 
                         start_id='start', tie_breaking='efron')
    
    return X, event_data, breslow, efron, W


def test_coxlm_breslow_comparison(sample_data):
    """Test CoxLM comparison with Breslow tie breaking."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxLM
    G2 = CoxLM(weight_id='weight', family=breslow, summarize=True)
    G2.fit(X, event_data)
    
    # R coxph
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {
        'event': FloatVector(event_numeric),
        'start': FloatVector(start_numeric),
        'status': FloatVector(status_numeric),
        'W': FloatVector(W_numeric)
    }
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create survival object and formula
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = survival.coxph(ro.Formula(formula_str), data=r_data, weights=FloatVector(W_numeric), 
                           ties='breslow', robust=False)
    r_coef = np.array(survival.coef(r_model))
    
    assert np.allclose(G2.coef_, r_coef)


def test_coxlm_efron_comparison(sample_data):
    """Test CoxLM comparison with Efron tie breaking."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxLM
    G3 = CoxLM(weight_id='weight', family=efron, summarize=True)
    G3.fit(X, event_data)
    
    # R coxph
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {
        'event': FloatVector(event_numeric),
        'start': FloatVector(start_numeric),
        'status': FloatVector(status_numeric),
        'W': FloatVector(W_numeric)
    }
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create survival object and formula
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = survival.coxph(ro.Formula(formula_str), data=r_data, weights=FloatVector(W_numeric), 
                           robust=False)
    r_coef = np.array(survival.coef(r_model))
    
    assert np.allclose(G3.coef_, r_coef, rtol=1e-4, atol=1e-4)


def test_coxnet_comparison(sample_data):
    """Test CoxNet comparison with R glmnet."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxNet
    GN = CoxNet(family=breslow, weight_id='weight')
    GN.fit(X, event_data)
    
    # R glmnet
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create survival object
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), 
                        r_surv, weights=FloatVector(W_numeric), family='cox')
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    # Compare results (using index 10 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[10], GN.coefs_[10], rtol=1e-4, atol=1e-4)


def test_cross_validation_fraction_alignment_grouped(sample_data):
    """Test cross-validation with fraction alignment (grouped)."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxNet with CV
    control = GLMNetControl()
    GN3 = CoxNet(family=breslow, weight_id='weight', control=control)
    GN3.fit(X, event_data)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X[:, :8], event_data, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create survival object
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    # Create the subset matrix X[,1:8] in R
    r_X_subset = numpy_to_r_matrix(X[:, :8])
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(r_X_subset, r_surv, weights=FloatVector(W_numeric), 
                             family='cox', foldid=r_foldid, alignment='fraction', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 10 as in original)
    assert np.allclose(GN3.cv_scores_['Cox Deviance (Difference)'].iloc[:10], r_cvm[:10], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], r_cvsd[:10], rtol=1e-3, atol=1e-3)


def test_cross_validation_lambda_alignment_grouped(sample_data):
    """Test cross-validation with lambda alignment (grouped)."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxNet with CV
    control = GLMNetControl()
    GN4 = CoxNet(family=breslow, weight_id='weight', control=control)
    GN4.fit(X, event_data)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X[:, :8], event_data, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create survival object
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X[:, :8]), 
                             r_surv, weights=FloatVector(W_numeric), family='cox', 
                             foldid=r_foldid, alignment='lambda', grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 10 as in original)
    assert np.allclose(GN4.cv_scores_['Cox Deviance (Difference)'].iloc[:10], r_cvm[:10], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], r_cvsd[:10], rtol=1e-3, atol=1e-3)


def test_cross_validation_fraction_alignment_ungrouped(sample_data):
    """Test cross-validation with fraction alignment (ungrouped)."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxNet with CV
    control = GLMNetControl()
    GN3 = CoxNet(family=breslow, weight_id='weight', control=control)
    GN3.fit(X, event_data)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X[:, :8], event_data, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create survival object
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X[:, :8]), 
                             r_surv, weights=FloatVector(W_numeric), family='cox', 
                             foldid=r_foldid, alignment='fraction', grouped=False)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 10 as in original)
    assert np.allclose(GN3.cv_scores_['Cox Deviance'].iloc[:10], r_cvm[:10], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10], r_cvsd[:10], rtol=1e-3, atol=1e-3)


def test_cross_validation_lambda_alignment_ungrouped(sample_data):
    """Test cross-validation with lambda alignment (ungrouped)."""
    X, event_data, breslow, efron, W = sample_data
    
    # Python CoxNet with CV
    control = GLMNetControl()
    GN4 = CoxNet(family=breslow, weight_id='weight', control=control)
    GN4.fit(X, event_data)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X[:, :8], event_data, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    event_numeric = event_data['event'].astype(float)
    start_numeric = event_data['start'].astype(float)
    status_numeric = event_data['status'].astype(float)
    W_numeric = W.astype(float)
    
    # Create survival object
    r_surv = survival.Surv(FloatVector(start_numeric), FloatVector(event_numeric), 
                          FloatVector(status_numeric))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X[:, :8]), 
                             r_surv, weights=FloatVector(W_numeric), family='cox', 
                             foldid=r_foldid, alignment='lambda', grouped=False)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 10 as in original)
    assert np.allclose(GN4.cv_scores_['Cox Deviance'].iloc[:10], r_cvm[:10], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.cv_scores_['SD(Cox Deviance)'].iloc[:10], r_cvsd[:10], rtol=1e-3, atol=1e-3) 
