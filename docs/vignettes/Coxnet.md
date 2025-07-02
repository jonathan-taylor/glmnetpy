---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
---

# Introduction

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. In the context of regularized regression, we can use the `glmnet` package to fit Cox models with various penalties (lasso, ridge, elastic net).

This vignette demonstrates how to use the Python `glmnet` package for regularized Cox regression, including handling of different data formats, cross-validation, and prediction.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Data Formats](#data-formats)
- [Cross-validation](#cross-validation)
- [Prediction](#prediction)
- [Advanced Features](#advanced-features)
- [References](#references)

---

# Installation

You can install the Python `glmnet` package using pip:

```{code-cell} ipython3
!pip install glmnetpy
```

---

# Basic Usage

Below is a quick demonstration of Cox regression using the Python `glmnet` API:

```{code-cell} ipython3
import numpy as np
import pandas as pd
from glmnet import CoxNet
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(1)

# Generate synthetic survival data
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [0.5, -0.3, 0.2, 0, -0.4]

# Generate survival times and event indicators
risk_scores = X @ beta
survival_times = np.random.exponential(scale=np.exp(-risk_scores))
events = np.random.binomial(1, 0.7, n)  # 70% event rate

# Fit Cox model
fit = CoxNet().fit(X, survival_times, events)

# Plot the coefficient paths
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Cox Regression: Regularization Paths")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

---

# Data Formats

The Cox model can handle different data formats depending on the type of survival analysis:

## Right-censored data

The most common format is right-censored data, where we have survival times and event indicators:

```{code-cell} ipython3
# Right-censored data format
# survival_times: time to event or censoring
# events: 1 for event, 0 for censoring

# Example with more realistic survival data
np.random.seed(42)
n, p = 200, 15

# Generate covariates
X = np.random.randn(n, p)
beta_true = np.zeros(p)
beta_true[:5] = [0.8, -0.6, 0.4, 0.2, -0.3]

# Generate survival times
risk_scores = X @ beta_true
baseline_hazard = 0.1
survival_times = np.random.exponential(scale=1/(baseline_hazard * np.exp(risk_scores)))

# Generate censoring times (administrative censoring)
censoring_times = np.random.exponential(scale=10, size=n)

# Determine observed times and events
observed_times = np.minimum(survival_times, censoring_times)
events = (survival_times <= censoring_times).astype(int)

print(f"Event rate: {events.mean():.2f}")
print(f"Mean survival time: {observed_times.mean():.2f}")

# Fit Cox model
fit = CoxNet().fit(X, observed_times, events)

# Plot coefficient paths
plt.figure(figsize=(12, 8))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Cox Regression: Coefficient Paths")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

## Start-stop data (time-dependent covariates)

For time-dependent covariates, we can use start-stop data format:

```{code-cell} ipython3
# Start-stop data format (for time-dependent covariates)
# This is more complex and typically requires specialized handling

# Example: Creating time-dependent data structure
# In practice, you would have data like:
# id, start_time, stop_time, event, covariate1, covariate2, ...

# For demonstration, we'll create a simple example
n_subjects = 50
n_visits = 3

# Create time-dependent data
data_list = []
for i in range(n_subjects):
    for j in range(n_visits):
        start_time = j * 2
        stop_time = (j + 1) * 2
        event = 1 if (i == n_subjects - 1 and j == n_visits - 1) else 0
        
        # Time-dependent covariates
        cov1 = np.random.randn()
        cov2 = np.random.randn()
        
        data_list.append({
            'id': i,
            'start': start_time,
            'stop': stop_time,
            'event': event,
            'cov1': cov1,
            'cov2': cov2
        })

# Convert to DataFrame
df = pd.DataFrame(data_list)
print("Time-dependent data structure:")
print(df.head(10))

# Note: The current Python glmnet implementation may not directly support
# start-stop data format. You may need to use specialized survival analysis
# packages like lifelines or statsmodels for this format.
```

---

# Cross-validation

Cross-validation is essential for selecting the optimal regularization parameter in Cox regression:

```{code-cell} ipython3
# Perform cross-validation
cvfit = CoxNet().fit(X, observed_times, events)
cv_results = cvfit.cv(X, observed_times, events, nfolds=5)

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.errorbar(np.log(cv_results['lambda']), cv_results['cvm'], 
             yerr=cv_results['cvup'] - cv_results['cvm'], 
             fmt='o-', capsize=3)
plt.axvline(np.log(cv_results['lambda_min']), color='red', linestyle='--', label='lambda.min')
plt.axvline(np.log(cv_results['lambda_1se']), color='blue', linestyle='--', label='lambda.1se')
plt.xlabel('log(lambda)')
plt.ylabel('Partial Likelihood Deviance')
plt.title('Cross-validation Results (Cox Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"lambda.min: {cv_results['lambda_min']:.4f}")
print(f"lambda.1se: {cv_results['lambda_1se']:.4f}")

# Get coefficients at optimal lambda
lambda_min_idx = np.argmin(np.abs(fit.lambda_values_ - cv_results['lambda_min']))
coef_min = fit.coefs_[lambda_min_idx]
print("\nCoefficients at lambda.min:")
for i, coef in enumerate(coef_min):
    if abs(coef) > 1e-6:
        print(f"Variable {i+1}: {coef:.4f}")
```

---

# Prediction

The Cox model can be used for prediction in several ways:

```{code-cell} ipython3
# Generate new data for prediction
np.random.seed(123)
new_X = np.random.randn(10, p)

# Make predictions at specific lambda values
lambda_vals = [0.1, 0.01]
for lam in lambda_vals:
    lambda_idx = np.argmin(np.abs(fit.lambda_values_ - lam))
    predictions = fit.predict(new_X, lambda_idx)
    print(f"Risk scores at lambda = {lam}:")
    print(predictions)
    print()

# Predict at optimal lambda from cross-validation
lambda_opt_idx = np.argmin(np.abs(fit.lambda_values_ - cv_results['lambda_min']))
risk_scores_opt = fit.predict(new_X, lambda_opt_idx)
print("Risk scores at optimal lambda:")
print(risk_scores_opt)
```

## Survival curves

We can also predict survival curves for new observations:

```{code-cell} ipython3
# For survival curve prediction, we would typically need:
# 1. Baseline hazard function
# 2. Time points for prediction
# 3. Individual risk scores

# This is a simplified example - in practice, you might use
# specialized survival analysis packages for detailed survival curves

# Calculate risk scores for all observations
risk_scores_all = fit.predict(X, lambda_opt_idx)

# Plot risk score distribution
plt.figure(figsize=(10, 6))
plt.hist(risk_scores_all, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.title('Distribution of Risk Scores')
plt.grid(True, alpha=0.3)
plt.show()

# Compare risk scores between events and non-events
plt.figure(figsize=(8, 6))
plt.boxplot([risk_scores_all[events == 0], risk_scores_all[events == 1]], 
            labels=['Censored', 'Events'])
plt.ylabel('Risk Score')
plt.title('Risk Scores by Event Status')
plt.grid(True, alpha=0.3)
plt.show()
```

---

# Advanced Features

## Stratified Cox regression

For stratified analysis, we can fit separate models for different strata:

```{code-cell} ipython3
# Create strata (e.g., based on age groups)
age_groups = np.random.choice(['young', 'middle', 'old'], size=n, p=[0.3, 0.4, 0.3])
strata = pd.Categorical(age_groups).codes

# Fit stratified models
strata_unique = np.unique(strata)
strata_models = {}

for s in strata_unique:
    mask = strata == s
    if mask.sum() > 10:  # Only fit if enough observations
        X_strata = X[mask]
        times_strata = observed_times[mask]
        events_strata = events[mask]
        
        fit_strata = CoxNet().fit(X_strata, times_strata, events_strata)
        strata_models[s] = fit_strata
        
        print(f"Stratum {s}: {mask.sum()} observations")

# Compare coefficients across strata
plt.figure(figsize=(15, 5))
for i, s in enumerate(strata_models.keys()):
    plt.subplot(1, len(strata_models), i+1)
    coefs = strata_models[s].coefs_[-1, :]  # Last lambda
    plt.bar(range(p), coefs)
    plt.title(f'Stratum {s}')
    plt.xlabel('Variable')
    plt.ylabel('Coefficient')
    plt.xticks(range(p), [f'V{j+1}' for j in range(p)], rotation=45)

plt.tight_layout()
plt.show()
```

## Penalty factors

We can apply different penalty factors to different variables:

```{code-cell} ipython3
# Set penalty factors: no penalty for first 3 variables
penalty_factor = np.ones(p)
penalty_factor[:3] = 0  # No penalty for first 3 variables

# Fit with custom penalty factors
fit_penalty = CoxNet(penalty_factor=penalty_factor).fit(X, observed_times, events)

# Plot results
plt.figure(figsize=(12, 8))
for i in range(p):
    plt.plot(np.log(fit_penalty.lambda_values_), fit_penalty.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Cox Regression with Custom Penalty Factors")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# Check that unpenalized variables stay in the model
print("Coefficients of unpenalized variables (should be non-zero):")
for i in range(3):
    print(f"Variable {i+1}: {fit_penalty.coefs_[-1, i]:.4f}")
```

## Elastic net mixing

We can use different values of alpha for elastic net:

```{code-cell} ipython3
# Compare different alpha values
alphas = [1.0, 0.5, 0.0]  # Lasso, Elastic Net, Ridge
models = {}

for alpha in alphas:
    fit_alpha = CoxNet(alpha=alpha).fit(X, observed_times, events)
    models[alpha] = fit_alpha

# Plot comparison
plt.figure(figsize=(15, 5))
for i, alpha in enumerate(alphas):
    plt.subplot(1, 3, i+1)
    coefs = models[alpha].coefs_[-1, :]  # Last lambda
    plt.bar(range(p), coefs)
    plt.title(f'Alpha = {alpha}')
    plt.xlabel('Variable')
    plt.ylabel('Coefficient')
    plt.xticks(range(p), [f'V{j+1}' for j in range(p)], rotation=45)

plt.tight_layout()
plt.show()
```

---

# Model Assessment

## Concordance index

We can assess model performance using the concordance index:

```{code-cell} ipython3
# Calculate concordance index
def calculate_concordance(risk_scores, times, events):
    """Calculate Harrell's C-index"""
    n = len(times)
    concordant = 0
    total = 0
    
    for i in range(n):
        for j in range(n):
            if i != j and events[i] == 1:  # Only consider actual events
                if times[i] < times[j]:
                    total += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif times[i] == times[j] and events[j] == 1:
                    total += 1
                    if risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
    
    return concordant / total if total > 0 else 0.5

# Calculate C-index for different lambda values
c_indices = []
lambda_indices = range(0, len(fit.lambda_values_), 5)  # Sample every 5th lambda

for i in lambda_indices:
    risk_scores = fit.predict(X, i)
    c_index = calculate_concordance(risk_scores, observed_times, events)
    c_indices.append(c_index)

# Plot C-index vs lambda
plt.figure(figsize=(10, 6))
plt.plot(np.log(fit.lambda_values_[lambda_indices]), c_indices, 'o-')
plt.xlabel('log(lambda)')
plt.ylabel('Concordance Index')
plt.title('Model Performance vs Regularization')
plt.grid(True, alpha=0.3)
plt.show()

# Find best lambda based on C-index
best_c_idx = np.argmax(c_indices)
best_lambda = fit.lambda_values_[lambda_indices[best_c_idx]]
print(f"Best lambda based on C-index: {best_lambda:.4f}")
print(f"Best C-index: {c_indices[best_c_idx]:.4f}")
```

## Time-dependent ROC

For time-dependent performance assessment:

```{code-cell} ipython3
# Calculate time-dependent ROC (simplified version)
# In practice, you might use specialized packages like lifelines

def calculate_td_roc(risk_scores, times, events, time_point):
    """Calculate time-dependent ROC at a specific time point"""
    # This is a simplified implementation
    # In practice, use specialized survival analysis packages
    
    # Create binary outcome at time_point
    binary_outcome = (times <= time_point) & (events == 1)
    
    # Calculate ROC
    from sklearn.metrics import roc_auc_score
    if len(np.unique(binary_outcome)) > 1:
        auc = roc_auc_score(binary_outcome, risk_scores)
        return auc
    else:
        return 0.5

# Calculate time-dependent AUC at different time points
time_points = [2, 4, 6, 8]
risk_scores_opt = fit.predict(X, lambda_opt_idx)

td_aucs = []
for t in time_points:
    auc = calculate_td_roc(risk_scores_opt, observed_times, events, t)
    td_aucs.append(auc)

plt.figure(figsize=(8, 6))
plt.plot(time_points, td_aucs, 'o-')
plt.xlabel('Time Point')
plt.ylabel('Time-dependent AUC')
plt.title('Time-dependent ROC Analysis')
plt.grid(True, alpha=0.3)
plt.show()
```

---

# References

1. Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5), 1-13.

2. Harrell, F. E., Lee, K. L., & Mark, D. B. (1996). Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Statistics in Medicine*, 15(4), 361-387.

3. Therneau, T. M., & Grambsch, P. M. (2000). *Modeling survival data: extending the Cox model*. Springer Science & Business Media.

---

*This document adapts the R glmnet Cox vignette for the Python glmnet package. The original R vignette was written by Trevor Hastie, Junyang Qian, and Kenneth Tay.*
