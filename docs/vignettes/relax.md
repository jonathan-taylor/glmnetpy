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

The relaxed lasso is a modification of the lasso that allows for a two-stage fitting process. In the first stage, the lasso is used to select variables, and in the second stage, the selected variables are refit without the lasso penalty. This approach can improve prediction accuracy in some cases.

This vignette demonstrates how to use the Python `glmnet` package for relaxed lasso fitting, including cross-validation and prediction.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
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

Below is a quick demonstration of relaxed lasso using the Python `glmnet` API:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glmnet import GaussNet

# Set random seed for reproducibility
np.random.seed(1)

# Generate synthetic data
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [1, 0.5, 0.5, 0, 2]
y = X @ beta + np.random.randn(n)

# Fit relaxed lasso
# Note: The Python glmnet implementation may not directly support relaxed lasso
# This is a conceptual demonstration of how it would work

# Stage 1: Lasso for variable selection
lasso_fit = GaussNet(alpha=1.0).fit(X, y)

# Get selected variables at a specific lambda
lambda_idx = 10  # Choose a lambda index
selected_vars = np.where(np.abs(lasso_fit.coefs_[lambda_idx]) > 1e-6)[0]
print(f"Selected variables at lambda index {lambda_idx}: {selected_vars}")

# Stage 2: Refit without penalty on selected variables
if len(selected_vars) > 0:
    X_selected = X[:, selected_vars]
    # Fit OLS on selected variables
    from sklearn.linear_model import LinearRegression
    ols_fit = LinearRegression().fit(X_selected, y)
    
    print("Relaxed lasso coefficients:")
    relaxed_coefs = np.zeros(p)
    relaxed_coefs[selected_vars] = ols_fit.coef_
    print(relaxed_coefs)
else:
    print("No variables selected by lasso")
```

---

# Cross-validation

Cross-validation for relaxed lasso involves selecting the optimal values for both lambda (for variable selection) and gamma (for the relaxation parameter):

```{code-cell} ipython3
# Perform cross-validation for relaxed lasso
# This is a conceptual implementation

def relaxed_lasso_cv(X, y, lambda_values, gamma_values, nfolds=5):
    """Cross-validation for relaxed lasso"""
    n = X.shape[0]
    fold_size = n // nfolds
    
    cv_results = []
    
    for lambda_val in lambda_values:
        for gamma_val in gamma_values:
            fold_errors = []
            
            for fold in range(nfolds):
                # Create fold indices
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < nfolds - 1 else n
                
                # Split data
                X_train = np.delete(X, slice(start_idx, end_idx), axis=0)
                y_train = np.delete(y, slice(start_idx, end_idx))
                X_test = X[start_idx:end_idx]
                y_test = y[start_idx:end_idx]
                
                # Stage 1: Lasso
                lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X_train, y_train)
                selected_vars = np.where(np.abs(lasso_fit.coefs_[0]) > 1e-6)[0]
                
                if len(selected_vars) > 0:
                    # Stage 2: Relaxed fit
                    X_train_selected = X_train[:, selected_vars]
                    X_test_selected = X_test[:, selected_vars]
                    
                    # Interpolate between lasso and OLS based on gamma
                    if gamma_val == 0:
                        # Pure lasso
                        pred = lasso_fit.predict(X_test, 0)
                    elif gamma_val == 1:
                        # Pure OLS
                        from sklearn.linear_model import LinearRegression
                        ols_fit = LinearRegression().fit(X_train_selected, y_train)
                        pred = ols_fit.predict(X_test_selected)
                    else:
                        # Interpolation
                        lasso_pred = lasso_fit.predict(X_test, 0)
                        from sklearn.linear_model import LinearRegression
                        ols_fit = LinearRegression().fit(X_train_selected, y_train)
                        ols_pred = ols_fit.predict(X_test_selected)
                        pred = (1 - gamma_val) * lasso_pred + gamma_val * ols_pred
                    
                    # Calculate MSE
                    mse = np.mean((y_test - pred) ** 2)
                    fold_errors.append(mse)
                else:
                    # No variables selected
                    fold_errors.append(np.mean(y_test ** 2))
            
            cv_results.append({
                'lambda': lambda_val,
                'gamma': gamma_val,
                'cvm': np.mean(fold_errors),
                'cvse': np.std(fold_errors) / np.sqrt(nfolds)
            })
    
    return cv_results

# Example usage
lambda_values = [0.1, 0.05, 0.01]
gamma_values = [0, 0.5, 1.0]

cv_results = relaxed_lasso_cv(X, y, lambda_values, gamma_values, nfolds=5)

# Find best parameters
best_idx = np.argmin([result['cvm'] for result in cv_results])
best_lambda = cv_results[best_idx]['lambda']
best_gamma = cv_results[best_idx]['gamma']

print(f"Best lambda: {best_lambda}")
print(f"Best gamma: {best_gamma}")
print(f"Best CV MSE: {cv_results[best_idx]['cvm']:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

# Plot CV results for each gamma
for gamma in gamma_values:
    gamma_results = [r for r in cv_results if r['gamma'] == gamma]
    lambdas = [r['lambda'] for r in gamma_results]
    cvms = [r['cvm'] for r in gamma_results]
    plt.plot(lambdas, cvms, 'o-', label=f'gamma={gamma}')

plt.xlabel('Lambda')
plt.ylabel('Cross-validation MSE')
plt.title('Relaxed Lasso Cross-validation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# Prediction

Prediction with relaxed lasso involves using the two-stage process:

```{code-cell} ipython3
# Prediction function for relaxed lasso
def relaxed_lasso_predict(X_new, X_train, y_train, lambda_val, gamma_val):
    """Make predictions using relaxed lasso"""
    
    # Stage 1: Lasso on training data
    lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X_train, y_train)
    selected_vars = np.where(np.abs(lasso_fit.coefs_[0]) > 1e-6)[0]
    
    if len(selected_vars) == 0:
        # No variables selected, return zero predictions
        return np.zeros(X_new.shape[0])
    
    # Stage 2: Relaxed fit
    X_train_selected = X_train[:, selected_vars]
    X_new_selected = X_new[:, selected_vars]
    
    if gamma_val == 0:
        # Pure lasso
        return lasso_fit.predict(X_new, 0)
    elif gamma_val == 1:
        # Pure OLS
        from sklearn.linear_model import LinearRegression
        ols_fit = LinearRegression().fit(X_train_selected, y_train)
        return ols_fit.predict(X_new_selected)
    else:
        # Interpolation
        lasso_pred = lasso_fit.predict(X_new, 0)
        from sklearn.linear_model import LinearRegression
        ols_fit = LinearRegression().fit(X_train_selected, y_train)
        ols_pred = ols_fit.predict(X_new_selected)
        return (1 - gamma_val) * lasso_pred + gamma_val * ols_pred

# Generate test data
np.random.seed(123)
X_test = np.random.randn(20, p)

# Make predictions with different gamma values
gamma_values = [0, 0.5, 1.0]
predictions = {}

for gamma in gamma_values:
    pred = relaxed_lasso_predict(X_test, X, y, best_lambda, gamma)
    predictions[f'gamma={gamma}'] = pred

# Compare predictions
plt.figure(figsize=(12, 4))
for i, (gamma, pred) in enumerate(predictions.items()):
    plt.subplot(1, 3, i+1)
    plt.hist(pred, bins=10, alpha=0.7, edgecolor='black')
    plt.title(f'Predictions: {gamma}')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Compare prediction ranges
print("Prediction ranges:")
for gamma, pred in predictions.items():
    print(f"{gamma}: [{pred.min():.3f}, {pred.max():.3f}]")
```

---

# Advanced Features

## Comparison with standard lasso

We can compare the performance of relaxed lasso with standard lasso:

```{code-cell} ipython3
# Compare relaxed lasso with standard lasso
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

# Standard lasso
lasso = Lasso(alpha=best_lambda)
lasso_scores = cross_val_score(lasso, X, y, cv=5, scoring='neg_mean_squared_error')
lasso_mse = -lasso_scores.mean()

# Relaxed lasso (using best parameters from CV)
def relaxed_lasso_scorer(estimator, X, y):
    """Custom scorer for relaxed lasso"""
    # This is a simplified implementation
    # In practice, you would need to implement the full relaxed lasso pipeline
    return -np.mean((y - estimator.predict(X)) ** 2)

# Fit relaxed lasso on full data
relaxed_pred = relaxed_lasso_predict(X, X, y, best_lambda, best_gamma)
relaxed_mse = np.mean((y - relaxed_pred) ** 2)

print(f"Standard Lasso CV MSE: {lasso_mse:.4f}")
print(f"Relaxed Lasso MSE: {relaxed_mse:.4f}")
print(f"Improvement: {(lasso_mse - relaxed_mse) / lasso_mse * 100:.2f}%")
```

## Variable selection stability

We can examine the stability of variable selection across different gamma values:

```{code-cell} ipython3
# Examine variable selection stability
lambda_val = 0.05
gamma_values = [0, 0.25, 0.5, 0.75, 1.0]

selection_stability = {}
for gamma in gamma_values:
    # Get selected variables
    lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
    selected_vars = np.where(np.abs(lasso_fit.coefs_[0]) > 1e-6)[0]
    selection_stability[f'gamma={gamma}'] = selected_vars

# Plot selection stability
plt.figure(figsize=(12, 6))
for i, (gamma, selected) in enumerate(selection_stability.items()):
    plt.subplot(1, len(gamma_values), i+1)
    selection_mask = np.zeros(p, dtype=bool)
    selection_mask[selected] = True
    plt.bar(range(p), selection_mask)
    plt.title(f'{gamma}')
    plt.xlabel('Variable')
    plt.ylabel('Selected')
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Print selection summary
print("Variable selection summary:")
for gamma, selected in selection_stability.items():
    print(f"{gamma}: {len(selected)} variables selected")
    if len(selected) > 0:
        print(f"  Variables: {selected}")
```

## Coefficient paths

We can visualize how coefficients change with different gamma values:

```{code-cell} ipython3
# Plot coefficient paths for different gamma values
lambda_val = 0.05
gamma_values = [0, 0.5, 1.0]

plt.figure(figsize=(15, 5))
for i, gamma in enumerate(gamma_values):
    plt.subplot(1, 3, i+1)
    
    # Get coefficients for this gamma
    if gamma == 0:
        # Pure lasso
        lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
        coefs = lasso_fit.coefs_[0]
    elif gamma == 1:
        # Pure OLS on selected variables
        lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
        selected_vars = np.where(np.abs(lasso_fit.coefs_[0]) > 1e-6)[0]
        if len(selected_vars) > 0:
            from sklearn.linear_model import LinearRegression
            ols_fit = LinearRegression().fit(X[:, selected_vars], y)
            coefs = np.zeros(p)
            coefs[selected_vars] = ols_fit.coef_
        else:
            coefs = np.zeros(p)
    else:
        # Interpolation
        lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
        lasso_coefs = lasso_fit.coefs_[0]
        selected_vars = np.where(np.abs(lasso_coefs) > 1e-6)[0]
        if len(selected_vars) > 0:
            from sklearn.linear_model import LinearRegression
            ols_fit = LinearRegression().fit(X[:, selected_vars], y)
            ols_coefs = np.zeros(p)
            ols_coefs[selected_vars] = ols_fit.coef_
            coefs = (1 - gamma) * lasso_coefs + gamma * ols_coefs
        else:
            coefs = np.zeros(p)
    
    plt.bar(range(p), coefs)
    plt.title(f'Coefficients: gamma={gamma}')
    plt.xlabel('Variable')
    plt.ylabel('Coefficient')
    plt.xticks(range(p), [f'V{j+1}' for j in range(p)], rotation=45)

plt.tight_layout()
plt.show()
```

---

# Model Interpretation

## Understanding the relaxation parameter

The gamma parameter controls the degree of relaxation:

- `gamma = 0`: Pure lasso (no relaxation)
- `gamma = 1`: Pure OLS on selected variables (full relaxation)
- `0 < gamma < 1`: Interpolation between lasso and OLS

```{code-cell} ipython3
# Demonstrate the effect of gamma
lambda_val = 0.05
gamma_values = np.linspace(0, 1, 11)

# Track how coefficients change with gamma
coef_trajectories = np.zeros((len(gamma_values), p))

for i, gamma in enumerate(gamma_values):
    # Get coefficients for this gamma
    lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
    lasso_coefs = lasso_fit.coefs_[0]
    selected_vars = np.where(np.abs(lasso_coefs) > 1e-6)[0]
    
    if len(selected_vars) > 0:
        from sklearn.linear_model import LinearRegression
        ols_fit = LinearRegression().fit(X[:, selected_vars], y)
        ols_coefs = np.zeros(p)
        ols_coefs[selected_vars] = ols_fit.coef_
        coefs = (1 - gamma) * lasso_coefs + gamma * ols_coefs
    else:
        coefs = np.zeros(p)
    
    coef_trajectories[i, :] = coefs

# Plot coefficient trajectories
plt.figure(figsize=(12, 8))
for j in range(p):
    plt.plot(gamma_values, coef_trajectories[:, j], 'o-', label=f'V{j+1}')

plt.xlabel('Gamma (relaxation parameter)')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Trajectories vs Gamma')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Show which variables are selected at different gamma values
plt.figure(figsize=(10, 6))
selection_matrix = (np.abs(coef_trajectories) > 1e-6).astype(int)
plt.imshow(selection_matrix.T, aspect='auto', cmap='Blues')
plt.xlabel('Gamma Index')
plt.ylabel('Variable')
plt.title('Variable Selection vs Gamma')
plt.colorbar(label='Selected (1) / Not Selected (0)')
plt.xticks(range(len(gamma_values)), [f'{g:.1f}' for g in gamma_values], rotation=45)
plt.yticks(range(p), [f'V{j+1}' for j in range(p)])
plt.tight_layout()
plt.show()
```

---

# Practical Considerations

## When to use relaxed lasso

Relaxed lasso can be beneficial when:

1. **High-dimensional data**: When p >> n, the lasso may be too conservative
2. **Correlated predictors**: When variables are highly correlated
3. **Prediction focus**: When the goal is prediction rather than variable selection

## Computational considerations

```{code-cell} ipython3
import time

# Compare computational time
lambda_val = 0.05
gamma_val = 0.5

# Time relaxed lasso
start_time = time.time()
relaxed_pred = relaxed_lasso_predict(X, X, y, lambda_val, gamma_val)
relaxed_time = time.time() - start_time

# Time standard lasso
start_time = time.time()
lasso_fit = GaussNet(alpha=1.0, lambda_values=[lambda_val]).fit(X, y)
lasso_pred = lasso_fit.predict(X, 0)
lasso_time = time.time() - start_time

print(f"Relaxed lasso time: {relaxed_time:.4f} seconds")
print(f"Standard lasso time: {lasso_time:.4f} seconds")
print(f"Time ratio: {relaxed_time/lasso_time:.2f}x")
```

## Model selection

For model selection in relaxed lasso, we need to choose both lambda and gamma:

```{code-cell} ipython3
# Grid search for optimal parameters
lambda_values = [0.1, 0.05, 0.01]
gamma_values = [0, 0.25, 0.5, 0.75, 1.0]

best_score = float('inf')
best_params = None

for lambda_val in lambda_values:
    for gamma_val in gamma_values:
        # Cross-validation score
        scores = []
        for fold in range(5):
            # Simple fold splitting
            fold_size = n // 5
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < 4 else n
            
            X_train = np.delete(X, slice(start_idx, end_idx), axis=0)
            y_train = np.delete(y, slice(start_idx, end_idx))
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            
            pred = relaxed_lasso_predict(X_test, X_train, y_train, lambda_val, gamma_val)
            mse = np.mean((y_test - pred) ** 2)
            scores.append(mse)
        
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = (lambda_val, gamma_val)

print(f"Best parameters: lambda={best_params[0]}, gamma={best_params[1]}")
print(f"Best CV score: {best_score:.4f}")
```

---

# References

1. Meinshausen, N. (2007). Relaxed lasso. *Computational Statistics & Data Analysis*, 52(1), 374-393.

2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical learning with sparsity: the lasso and generalizations*. CRC press.

3. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

---

*This document adapts the R glmnet relax vignette for the Python glmnet package. The original R vignette was written by Trevor Hastie, Junyang Qian, and Kenneth Tay.*
