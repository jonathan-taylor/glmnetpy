---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Introduction

`glmnet` is a package that fits generalized linear and similar models via penalized maximum likelihood. The regularization path is computed for the lasso or elastic net penalty at a grid of values (on the log scale) for the regularization parameter lambda. The algorithm is extremely fast, and can exploit sparsity in the input matrix `X`. It fits linear, logistic and multinomial, poisson, and Cox regression models. It can also fit multi-response linear regression, generalized linear models for custom families, and relaxed lasso regression models. The package includes methods for prediction and plotting, and functions for cross-validation.

The original authors of glmnet are Jerome Friedman, Trevor Hastie, Rob Tibshirani, Balasubramanian Narasimhan, Kenneth Tay and Noah Simon, with contribution from Junyang Qian. This document adapts the R vignette for the Python `glmnet` package.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Linear Regression: family = "gaussian"](#linear-regression-family--gaussian-)
- [Linear Regression: family = "mgaussian"](#linear-regression-family--mgaussian--multi-response)
- [Logistic Regression: family = "binomial"](#logistic-regression-family--binomial-)
- [Multinomial Regression: family = "multinomial"](#multinomial-regression-family--multinomial-)
- [Poisson Regression: family = "poisson"](#poisson-regression-family--poisson-)
- [Cox Regression: family = "cox"](#cox-regression-family--cox-)
- [Programmable GLM families](#programmable-glm-families-family--family-)
- [Assessing models on test data](#assessing-models-on-test-data)
- [Filtering variables](#filtering-variables)
- [Other Package Features](#other-package-features)
- [Appendix: Convergence Criteria and Internal Parameters](#appendix-0-convergence-criteria)
- [References](#references)

---

# Installation

You can install the Python `glmnet` package using pip:

```{code-cell} ipython3
!pip install glmnetpy
```

---

# Quick Start

Below is a quick demonstration of the main functions and outputs using the Python `glmnet` API. We will use synthetic data for illustration.

```{code-cell} ipython3
import numpy as np
from glmnet import LogNet, GaussNet, GLMNet
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(1)

# Generate synthetic data
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [1, 0.5, 0.5, 0, 2]
y = X @ beta + np.random.randn(n)

# Fit a Gaussian model (linear regression)
fit = GaussNet().fit(X, y)

# Plot the coefficient paths
plt.figure(figsize=(8, 5))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:,i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths (Lasso)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

---

# Linear Regression: family = "gaussian"

The default model used in the package is the Gaussian linear model or "least squares". We'll demonstrate this using synthetic data for illustration:

```{code-cell} ipython3
import numpy as np
from glmnet import GaussNet
import matplotlib.pyplot as plt

# Generate synthetic data similar to QuickStartExample
np.random.seed(1)
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [1, 0.5, 0.5, 0, 2]
y = X @ beta + np.random.randn(n)

# Fit the model using the most basic call
fit = GaussNet().fit(X, y)
```

`fit` is an object that contains all the relevant information of the fitted model for further use. We can visualize the coefficients by plotting the regularization paths:

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths (Lasso)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

Each curve corresponds to a variable. It shows the path of its coefficient against the $\ell_1$-norm of the whole coefficient vector as $\lambda$ varies.

We can obtain the model coefficients at specific $\lambda$ values:

```{code-cell} ipython3
# Get coefficients at lambda = 0.1
lambda_idx = np.argmin(np.abs(fit.lambda_values_ - 0.1))
print("Coefficients at lambda = 0.1:")
print(fit.coefs_[lambda_idx])
```

Users can also make predictions at specific $\lambda$ values with new input data:

```{code-cell} ipython3
# Generate new data for prediction
np.random.seed(29)
new_X = np.random.randn(5, 20)

# Make predictions at specific lambda values
lambda_vals = [0.1, 0.05]
for lam in lambda_vals:
    lambda_idx = np.argmin(np.abs(fit.lambda_values_ - lam))
    predictions = fit.predict(new_X, lambda_idx)
    print(f"Predictions at lambda = {lam}:")
    print(predictions)
    print()
```

## Cross-validation

Cross-validation is perhaps the simplest and most widely used method for selecting the optimal $\lambda$. We can perform cross-validation using the `cv` method:

```{code-cell} ipython3
from glmnet import GaussNet

# Perform cross-validation
cvfit = GaussNet().fit(X, y)
cv_results = cvfit.cv(X, y, nfolds=10)

# Plot cross-validation results
plt.figure(figsize=(8, 6))
plt.errorbar(np.log(cv_results['lambda']), cv_results['cvm'], 
             yerr=cv_results['cvup'] - cv_results['cvm'], 
             fmt='o-', capsize=3)
plt.axvline(np.log(cv_results['lambda_min']), color='red', linestyle='--', label='lambda.min')
plt.axvline(np.log(cv_results['lambda_1se']), color='blue', linestyle='--', label='lambda.1se')
plt.xlabel('log(lambda)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-validation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"lambda.min: {cv_results['lambda_min']:.4f}")
print(f"lambda.1se: {cv_results['lambda_1se']:.4f}")
```

The cross-validation curve shows the mean squared error (red dotted line) along with upper and lower standard deviation curves. Two special values are indicated:

- `lambda.min`: the value of $\lambda$ that gives minimum mean cross-validated error
- `lambda.1se`: the value of $\lambda$ that gives the most regularized model such that the cross-validated error is within one standard error of the minimum

We can get the model coefficients at these optimal values:

```{code-cell} ipython3
# Get coefficients at lambda.min
lambda_min_idx = np.argmin(np.abs(fit.lambda_values_ - cv_results['lambda_min']))
coef_min = fit.coefs_[lambda_min_idx]
print("Coefficients at lambda.min:")
print(coef_min)
```

## Commonly used function arguments

`GaussNet` provides various arguments for users to customize the fit:

- `alpha`: the elastic net mixing parameter $\alpha$, with range $\alpha \in [0,1]$. $\alpha = 1$ is lasso regression (default) and $\alpha = 0$ is ridge regression.

- `weights`: observation weights, default is 1 for each observation.

- `nlambda`: the number of $\lambda$ values in the sequence (default is 100).

- `lambda_values`: can be provided if the user wants to specify the lambda sequence.

- `standardize`: logical flag for $x$ variable standardization prior to fitting the model sequence.

As an example, we set $\alpha = 0.2$ (more like a ridge regression), and give double weight to the latter half of the observations:

```{code-cell} ipython3
# Create weights: double weight for latter half
weights = np.ones(n)
weights[n//2:] = 2

# Fit with alpha=0.2 and custom weights
fit_ridge = GaussNet(alpha=0.2, nlambda=20).fit(X, y, sample_weight=weights)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit_ridge.lambda_values_), fit_ridge.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths (Elastic Net, α=0.2)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

## Predicting and plotting with fitted objects

We can extract the coefficients and make predictions for a fitted object at certain values of $\lambda$. Here is a simple example:

```{code-cell} ipython3
# Check if 0.5 is in the original lambda sequence
lambda_05_idx = np.where(np.abs(fit.lambda_values_ - 0.5) < 1e-10)[0]
if len(lambda_05_idx) > 0:
    print("0.5 is in the original lambda sequence")
    coef_exact = fit.coefs_[lambda_05_idx[0]]
else:
    print("0.5 is not in the original lambda sequence")
    # Find closest lambda
    closest_idx = np.argmin(np.abs(fit.lambda_values_ - 0.5))
    coef_approx = fit.coefs_[closest_idx]
    print(f"Using closest lambda: {fit.lambda_values_[closest_idx]:.4f}")
    print("Approximate coefficients:")
    print(coef_approx)
```

Users can make predictions from the fitted object. The `predict` method allows users to choose the type of prediction returned:

- "link" returns the fitted values (i.e. $\hat\beta_0 + x_i^T\hat\beta$)
- "response" gives the same output as "link" for "gaussian" family
- "coefficients" returns the model coefficients

For example, the following code gives the fitted values for the first 5 observations at $\lambda = 0.05$:

```{code-cell} ipython3
lambda_idx = np.argmin(np.abs(fit.lambda_values_ - 0.05))
predictions = fit.predict(X[:5], lambda_idx)
print("Fitted values for first 5 observations at λ = 0.05:")
print(predictions)
```

## Other function arguments

In this section we briefly describe some other useful arguments when calling `GaussNet`: `upper_limits`, `lower_limits`, `penalty_factor`, and `intercept`.

Suppose we want to fit our model but limit the coefficients to be bigger than -0.7 and less than 0.5:

```{code-cell} ipython3
# Fit with coefficient limits
fit_limited = GaussNet(lower_limits=-0.7, upper_limits=0.5).fit(X, y)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit_limited.lambda_values_), fit_limited.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths with Coefficient Limits")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

Often we want the coefficients to be positive: to do so, we just need to specify `lower_limits = 0`:

```{code-cell} ipython3
# Fit with non-negative coefficients
fit_positive = GaussNet(lower_limits=0).fit(X, y)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit_positive.lambda_values_), fit_positive.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths (Non-negative Coefficients)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

The `penalty_factor` argument allows users to apply separate penalty factors to each coefficient. This is very useful when we have prior knowledge or preference over the variables:

```{code-cell} ipython3
# Set penalty factors for variables 1, 3, and 5 to be zero (no penalty)
penalty_factor = np.ones(p)
penalty_factor[[0, 2, 4]] = 0  # Variables 1, 3, 5 (0-indexed)

fit_penalty = GaussNet(penalty_factor=penalty_factor).fit(X, y)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit_penalty.lambda_values_), fit_penalty.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths with Custom Penalty Factors")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

We see from the plot that the three variables with zero penalty factors always stay in the model, while the others follow typical regularization paths and are shrunk to zero eventually.

The `intercept` argument allows the user to decide if an intercept should be included in the model or not (it is never penalized). The default is `intercept = True`:

```{code-cell} ipython3
# Fit without intercept
fit_no_intercept = GaussNet(intercept=False).fit(X, y)

# Compare with intercept
fit_with_intercept = GaussNet(intercept=True).fit(X, y)

print("Intercept values:")
print(f"With intercept: {fit_with_intercept.intercept_[0]:.4f}")
print(f"Without intercept: {fit_no_intercept.intercept_[0]:.4f}")
```

# Linear Regression: family = "mgaussian" (multi-response)

The multi-response Gaussian family is useful when there are a number of (correlated) responses, also known as the "multi-task learning" problem. Here, a variable is either included in the model for all the responses, or excluded for all the responses.

As the name suggests, the response $y$ is not a vector but a matrix of quantitative responses. As a result, the coefficients at each value of lambda are also a matrix.

`glmnet` solves the problem
$$
\min_{(\beta_0, \beta) \in \mathbb{R}^{(p+1)\times K}}\frac{1}{2N} \sum_{i=1}^N \|y_i -\beta_0-\beta^T x_i\|^2_F+\lambda \left[ (1-\alpha)\|\beta\|_F^2/2 + \alpha\sum_{j=1}^p\|\beta_j\|_2\right].
$$
Here $\beta_j$ is the $j$th row of the $p\times K$ coefficient matrix $\beta$, and we replace the absolute penalty on each single coefficient by a group-lasso penalty on each coefficient $K$-vector $\beta_j$ for a single predictor.

We use synthetic data for illustration:

```{code-cell} ipython3
import numpy as np
from glmnet import MultiGaussNet
import matplotlib.pyplot as plt

# Generate multi-response data
np.random.seed(1)
n, p, K = 100, 20, 3
X = np.random.randn(n, p)
beta = np.zeros((p, K))
beta[:5, :] = np.random.randn(5, K)  # First 5 variables are active
y = X @ beta + np.random.randn(n, K)

# Fit a regularized multi-response Gaussian model
mfit = MultiGaussNet().fit(X, y)
```

We can visualize the coefficients by plotting the $\ell_2$ norm of each variable's coefficient vector:

```{code-cell} ipython3
# Calculate L2 norm of coefficients for each variable
coef_norms = np.linalg.norm(mfit.coefs_, axis=2)  # Shape: (n_lambda, p)

plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(mfit.lambda_values_), coef_norms[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("L2 norm of coefficients")
plt.title("Multi-response Gaussian: Coefficient Norms")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

We can extract the coefficients and make predictions at requested values of $\lambda$:

```{code-cell} ipython3
# Make predictions at specific lambda values
lambda_vals = [0.1, 0.01]
for lam in lambda_vals:
    lambda_idx = np.argmin(np.abs(mfit.lambda_values_ - lam))
    predictions = mfit.predict(X[:5], lambda_idx)
    print(f"Predictions at lambda = {lam}:")
    print(f"Shape: {predictions.shape}")
    print(predictions)
    print()
```

# Logistic Regression: family = "binomial"

Logistic regression is a widely-used model when the response is binary. Suppose the response variable takes values in $\mathcal{G}=\{1,2\}$. Denote $y_i = I(g_i=1)$. We model

$$\mbox{Pr}(G=2|X=x)=\frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}},$$

which can be written in the following form:

$$\log\frac{\mbox{Pr}(G=2|X=x)}{\mbox{Pr}(G=1|X=x)}=\beta_0+\beta^Tx,$$
the so-called "logistic" or log-odds transformation.

The objective function for logistic regression is the penalized negative binomial log-likelihood:
$$
\min_{(\beta_0, \beta) \in \mathbb{R}^{p+1}} -\left[\frac{1}{N} \sum_{i=1}^N y_i \cdot (\beta_0 + x_i^T \beta) - \log (1+e^{(\beta_0+x_i^T \beta)})\right] + \lambda \big[ (1-\alpha)\|\beta\|_2^2/2 + \alpha\|\beta\|_1\big].
$$

For illustration purposes, we generate synthetic data:

```{code-cell} ipython3
import numpy as np
from glmnet import LogNet
import matplotlib.pyplot as plt

# Generate binary classification data
np.random.seed(1)
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [1, 0.5, 0.5, 0, 2]
logits = X @ beta
probs = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probs)

# Fit logistic regression
fit = LogNet().fit(X, y)
```

As before, we can print and plot the fitted object, extract the coefficients at specific $\lambda$'s and also make predictions:

```{code-cell} ipython3
# Plot coefficient paths
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression: Regularization Paths")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

Prediction is a little different for logistic regression, mainly in the function argument `type`:

- "link" gives the linear predictors
- "response" gives the fitted probabilities
- "class" produces the class label corresponding to the maximum probability

```{code-cell} ipython3
# Make predictions of different types
lambda_idx = np.argmin(np.abs(fit.lambda_values_ - 0.05))

# Linear predictors
link_pred = fit.predict(X[:5], lambda_idx, type="link")
print("Linear predictors (link):")
print(link_pred)

# Probabilities
prob_pred = fit.predict(X[:5], lambda_idx, type="response")
print("\nFitted probabilities (response):")
print(prob_pred)

# Class predictions
class_pred = fit.predict(X[:5], lambda_idx, type="class")
print("\nClass predictions:")
print(class_pred)
```

For logistic regression, cross-validation has similar arguments and usage as Gaussian:

```{code-cell} ipython3
# Perform cross-validation
cvfit = LogNet().fit(X, y)
cv_results = cvfit.cv(X, y, nfolds=10, type_measure="class")

# Plot cross-validation results
plt.figure(figsize=(8, 6))
plt.errorbar(np.log(cv_results['lambda']), cv_results['cvm'], 
             yerr=cv_results['cvup'] - cv_results['cvm'], 
             fmt='o-', capsize=3)
plt.axvline(np.log(cv_results['lambda_min']), color='red', linestyle='--', label='lambda.min')
plt.axvline(np.log(cv_results['lambda_1se']), color='blue', linestyle='--', label='lambda.1se')
plt.xlabel('log(lambda)')
plt.ylabel('Misclassification Error')
plt.title('Cross-validation Results (Logistic Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"lambda.min: {cv_results['lambda_min']:.4f}")
print(f"lambda.1se: {cv_results['lambda_1se']:.4f}")
```

# Multinomial Regression: family = "multinomial"

The multinomial model extends the binomial when the number of classes is more than two. Suppose the response variable has $K$ levels ${\cal G}=\{1,2,\ldots,K\}$. Here we model
$$\mbox{Pr}(G=k|X=x)=\frac{e^{\beta_{0k}+\beta_k^Tx}}{\sum_{\ell=1}^Ke^{\beta_{0\ell}+\beta_\ell^Tx}}.$$
There is a linear predictor for each class!

Let ${Y}$ be the $N \times K$ indicator response matrix, with elements $y_{i\ell} = I(g_i=\ell)$. Then the elastic net penalized negative log-likelihood function becomes
$$
\ell(\{\beta_{0k},\beta_{k}\}_1^K) = -\left[\frac{1}{N} \sum_{i=1}^N \Big(\sum_{k=1}^Ky_{il} (\beta_{0k} + x_i^T \beta_k)- \log \big(\sum_{\ell=1}^K e^{\beta_{0\ell}+x_i^T \beta_\ell}\big)\Big)\right] +\lambda \left[ (1-\alpha)\|\beta\|_F^2/2 + \alpha\sum_{j=1}^p\|\beta_j\|_q\right].
$$

We support two options for $q$: $q\in \{1,2\}$. When $q=1$, this is a lasso penalty on each of the parameters. When $q=2$, this is a grouped-lasso penalty on all the $K$ coefficients for a particular variable, which makes them all be zero or nonzero together.

For the `family = "multinomial"` case, usage is similar to that for `family = "binomial"`. We generate synthetic data:

```{code-cell} ipython3
import numpy as np
from glmnet import MultiClassNet
import matplotlib.pyplot as plt

# Generate multinomial data
np.random.seed(1)
n, p, K = 100, 20, 3
X = np.random.randn(n, p)
beta = np.zeros((p, K))
beta[:5, :] = np.random.randn(5, K)  # First 5 variables are active

# Generate class probabilities
logits = X @ beta
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
y = np.array([np.random.choice(K, p=prob) for prob in probs])

# Fit multinomial regression with grouped lasso penalty
fit = MultiClassNet(type_multinomial="grouped").fit(X, y)
```

For the `plot` method, we can produce a figure showing the $\ell_2$-norm in one figure:

```{code-cell} ipython3
# Calculate L2 norm of coefficients for each variable
coef_norms = np.linalg.norm(fit.coefs_, axis=2)  # Shape: (n_lambda, p)

plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), coef_norms[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("L2 norm of coefficients")
plt.title("Multinomial Regression: Coefficient Norms (Grouped)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

We can also do cross-validation:

```{code-cell} ipython3
# Perform cross-validation
cvfit = MultiClassNet(type_multinomial="grouped").fit(X, y)
cv_results = cvfit.cv(X, y, nfolds=10)

# Plot cross-validation results
plt.figure(figsize=(8, 6))
plt.errorbar(np.log(cv_results['lambda']), cv_results['cvm'], 
             yerr=cv_results['cvup'] - cv_results['cvm'], 
             fmt='o-', capsize=3)
plt.axvline(np.log(cv_results['lambda_min']), color='red', linestyle='--', label='lambda.min')
plt.axvline(np.log(cv_results['lambda_1se']), color='blue', linestyle='--', label='lambda.1se')
plt.xlabel('log(lambda)')
plt.ylabel('Cross-validation Error')
plt.title('Cross-validation Results (Multinomial)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Users may wish to predict at the optimally selected $\lambda$:

```{code-cell} ipython3
# Make predictions at lambda.min
lambda_min_idx = np.argmin(np.abs(fit.lambda_values_ - cv_results['lambda_min']))
predictions = fit.predict(X[:10], lambda_min_idx, type="class")
print("Class predictions at lambda.min:")
print(predictions)
```

# Poisson Regression: family = "poisson"

Poisson regression is used to model count data under the assumption of Poisson error, or otherwise non-negative data where the mean and variance are proportional. Like the Gaussian and binomial models, the Poisson distribution is a member of the exponential family of distributions. We usually model its positive mean on the log scale: $\log \mu(x) = \beta_0+\beta' x$.

The log-likelihood for observations $\{x_i,y_i\}_1^N$ is given by
$$
l(\beta|X, Y) = \sum_{i=1}^N \left(y_i (\beta_0+\beta^T x_i) - e^{\beta_0+\beta^Tx_i}\right).
$$
As before, we optimize the penalized log-likelihood:
$$
\min_{\beta_0,\beta} -\frac1N l(\beta|X, Y)  + \lambda \left((1-\alpha) \sum_{i=1}^N \beta_i^2/2 +\alpha \sum_{i=1}^N |\beta_i|\right).
$$

We generate synthetic Poisson data:

```{code-cell} ipython3
import numpy as np
from glmnet import FishNet
import matplotlib.pyplot as plt

# Generate Poisson data
np.random.seed(1)
n, p = 100, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [0.5, 0.3, 0.2, 0, 0.8]
log_means = X @ beta
means = np.exp(log_means)
y = np.random.poisson(means)

# Fit Poisson regression
fit = FishNet().fit(X, y)
```

We plot the coefficients to have a first sense of the result:

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(np.log(fit.lambda_values_), fit.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Poisson Regression: Regularization Paths")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
```

As before, we can extract the coefficients and make predictions at certain $\lambda$'s:

```{code-cell} ipython3
# Get coefficients at lambda = 1
lambda_idx = np.argmin(np.abs(fit.lambda_values_ - 1))
coef_1 = fit.coefs_[lambda_idx]
print("Coefficients at lambda = 1:")
print(coef_1)

# Make predictions
lambda_vals = [0.1, 1]
for lam in lambda_vals:
    lambda_idx = np.argmin(np.abs(fit.lambda_values_ - lam))
    predictions = fit.predict(X[:5], lambda_idx, type="response")
    print(f"\nPredictions at lambda = {lam}:")
    print(predictions)
```

We may also use cross-validation to find the optimal $\lambda$'s:

```{code-cell} ipython3
# Perform cross-validation
cvfit = FishNet().fit(X, y)
cv_results = cvfit.cv(X, y, nfolds=10)

# Plot cross-validation results
plt.figure(figsize=(8, 6))
plt.errorbar(np.log(cv_results['lambda']), cv_results['cvm'], 
             yerr=cv_results['cvup'] - cv_results['cvm'], 
             fmt='o-', capsize=3)
plt.axvline(np.log(cv_results['lambda_min']), color='red', linestyle='--', label='lambda.min')
plt.axvline(np.log(cv_results['lambda_1se']), color='blue', linestyle='--', label='lambda.1se')
plt.xlabel('log(lambda)')
plt.ylabel('Deviance')
plt.title('Cross-validation Results (Poisson)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

# Cox Regression: family = "cox"

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. We have a separate vignette dedicated solely to fitting regularized Cox models with the `glmnet` package; please consult that vignette for details.

# Programmable GLM families: family = family()

Since version 4.0, `glmnet` has the facility to fit any GLM family by specifying a `family` object, as used by `stats::glm`. For these more general families, the outer Newton loop is performed in R, while the inner elastic-net loop is performed in Fortran, for each value of lambda. The price for this generality is a small hit in speed.

For details, see the vignette "GLM `family` functions in `glmnet`".

# Assessing models on test data

Once we have fit a series of models using `glmnet`, we often assess their performance on a set of evaluation or test data. We usually go through the process of building a prediction matrix, deciding on the performance measure, and computing these measures for a series of values for `lambda`.

## Performance measures

We can compute performance measures on a validation or test dataset. Here's an example for logistic regression:

```{code-cell} ipython3
# Split data into train and test
np.random.seed(1)
n_train = 70
train_idx = np.random.choice(n, n_train, replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Fit model on training data
fit = LogNet(nlambda=5).fit(X_train, y_train)

# Make predictions on test data
predictions = fit.predict(X_test)

# Calculate performance measures
from sklearn.metrics import accuracy_score, log_loss

# For each lambda, calculate accuracy and log loss
lambda_indices = range(len(fit.lambda_values_))
accuracies = []
log_losses = []

for i in lambda_indices:
    pred_probs = fit.predict(X_test, i, type="response")
    pred_classes = fit.predict(X_test, i, type="class")
    
    acc = accuracy_score(y_test, pred_classes)
    ll = log_loss(y_test, pred_probs)
    
    accuracies.append(acc)
    log_losses.append(ll)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(np.log(fit.lambda_values_), accuracies, 'o-')
ax1.set_xlabel('log(lambda)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Test Accuracy')
ax1.grid(True, alpha=0.3)

ax2.plot(np.log(fit.lambda_values_), log_losses, 'o-')
ax2.set_xlabel('log(lambda)')
ax2.set_ylabel('Log Loss')
ax2.set_title('Test Log Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find best lambda based on accuracy
best_idx = np.argmax(accuracies)
print(f"Best lambda for accuracy: {fit.lambda_values_[best_idx]:.4f}")
print(f"Best accuracy: {accuracies[best_idx]:.4f}")
```

## ROC curves for binomial data

In the special case of binomial models, users often would like to see the ROC curve for validation or test data:

```{code-cell} ipython3
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve for the best lambda
best_lambda_idx = np.argmax(accuracies)
pred_probs = fit.predict(X_test, best_lambda_idx, type="response")

fpr, tpr, _ = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
```

# Filtering variables

Sometimes we want to filter variables before fitting the model. This can be done using the `exclude` argument:

```{code-cell} ipython3
# Exclude variables 1, 3, and 5 from the model
exclude_vars = [0, 2, 4]  # 0-indexed
fit_excluded = GaussNet().fit(X, y, exclude=exclude_vars)

# Plot results
plt.figure(figsize=(10, 6))
for i in range(p):
    if i not in exclude_vars:
        plt.plot(np.log(fit_excluded.lambda_values_), fit_excluded.coefs_[:, i], label=f"V{i+1}")
plt.xlabel("log(lambda)")
plt.ylabel("Coefficients")
plt.title("Regularization Paths (Excluded Variables 1, 3, 5)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# Check that excluded variables have zero coefficients
print("Coefficients of excluded variables (should be zero):")
for i in exclude_vars:
    print(f"Variable {i+1}: {fit_excluded.coefs_[-1, i]}")
```

# Other Package Features

## Offsets

Like other generalized linear models, `glmnet` allows for an "offset". This is a fixed vector of $N$ numbers that is added into the linear predictor. For example, you may have fitted some other logistic regression using other variables (and data), and now you want to see if the present variables can add further predictive power. To do this, you can use the predicted logit from the other model as an offset in the `glmnet` call.

```{code-cell} ipython3
# Example with offset in logistic regression
# Generate offset (e.g., from another model)
offset = np.random.randn(n)

# Fit with offset
fit_with_offset = LogNet().fit(X, y, offset=offset)

# Compare with fit without offset
fit_no_offset = LogNet().fit(X, y)

print("Intercept with offset:", fit_with_offset.intercept_[0])
print("Intercept without offset:", fit_no_offset.intercept_[0])
```

## Parallel computing

For large-scale problems, parallel computing can significantly speed up the computation process. The Python `glmnet` package supports parallel processing through joblib:

```{code-cell} ipython3
# Example of parallel cross-validation
from joblib import parallel_backend

# This would be used in practice for large datasets
# with parallel_backend('threading', n_jobs=4):
#     cv_results = fit.cv(X, y, nfolds=10)
```

# Appendix: Convergence Criteria and Internal Parameters

The `glmnet` algorithm uses several internal parameters to control convergence and numerical stability:

- **Tolerance**: The algorithm stops when the relative change in the objective function is less than a specified tolerance.
- **Maximum iterations**: The maximum number of iterations for the coordinate descent algorithm.
- **Warm starts**: The algorithm uses warm starts to efficiently compute the entire regularization path.

These parameters can be adjusted if needed, though the defaults work well for most applications.

# References

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical learning with sparsity: the lasso and generalizations*. CRC press.

3. Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5), 1-13.

---

*This document adapts the R glmnet vignette for the Python glmnet package. The original R vignette was written by Trevor Hastie, Junyang Qian, and Kenneth Tay.*
