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

`glmnet` is a Python package that fits generalized linear models via penalized maximum likelihood. The regularization path is computed for the lasso or elastic net penalty at a grid of values (on the log scale) for the regularization parameter lambda. The algorithm is extremely fast and can exploit sparsity in the input matrix `X`. It fits linear, logistic and multinomial, poisson, and Cox regression models. It can also fit multi-response linear regression, generalized linear models for custom families, and relaxed lasso regression models. The package includes methods for prediction and plotting, and functions for cross-validation.

The original authors of glmnet are Jerome Friedman, Trevor Hastie, Rob Tibshirani, Balasubramanian Narasimhan, Kenneth Tay and Noah Simon, with contribution from Junyang Qian. This document adapts the R vignette for the Python `glmnet` package.

# Installation

You can install the Python `glmnet` package using pip:

```{code-cell} ipython3
#!pip install glmnetpy
```

---

# Quick Start

Below is a quick demonstration of the main functions and outputs using the Python `glmnet` API. We will use synthetic data for illustration.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from glmnet import GaussNet

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic regression data
X_gaussian, y_gaussian, coef = make_regression(n_samples=100, n_features=20, n_informative=5, 
                            noise=3.0, coef=True, random_state=42)

# Fit a Gaussian model (linear regression)
fit = GaussNet().fit(X_gaussian, y_gaussian)

# Plot the coefficient paths
fit.coef_path_.plot();
plt.title('Coefficient Paths for Linear Regression');
```

---

# Linear Regression: GaussNet

The default model used in the package is the Gaussian linear model or "least squares". We'll demonstrate this using synthetic data for illustration:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from glmnet import GaussNet

# Generate synthetic regression data
X_gaussian, y_gaussian, coef = make_regression(n_samples=100, n_features=20, n_informative=5, 
                             noise=3.0, coef=True, random_state=42)

# Fit the model using the most basic call
fit = GaussNet().fit(X_gaussian, y_gaussian)
```

The fitted object contains all the relevant information of the fitted model for further use. We can visualize the coefficients by plotting the regularization paths:

```{code-cell} ipython3
fit.coef_path_.plot();
plt.title('Coefficient Paths for Linear Regression');
```

Each curve corresponds to a variable. It shows the path of its coefficient against the $\ell_1$-norm of the whole coefficient vector as $\lambda$ varies.

We can obtain the model coefficients at specific $\lambda$ values by interpolation:

```{code-cell} ipython3
# Get coefficients at lambda = 0.1
coefs, intercept = fit.interpolate_coefs(0.1)
print("Coefficients at lambda = 0.1:")
print(coefs)
```

Users can also make predictions at specific $\lambda$ values with new input data:

```{code-cell} ipython3
# Generate new data for prediction
from sklearn.datasets import make_regression
new_X, _ = make_regression(n_samples=5, n_features=20, n_informative=5, 
                             noise=3.0, random_state=123)

# Make predictions at specific lambda values
lambda_values = [0.1, 0.05]
predictions = fit.predict(new_X, interpolation_grid=lambda_values)
print(f"Prediction shape: {predictions.shape}")
```

## Cross-validation

Cross-validation is perhaps the simplest and most widely used method for selecting the optimal $\lambda$. We can perform cross-validation using the `cross_validation_path` method:

```{code-cell} ipython3
from glmnet import GaussNet

# Perform cross-validation
cvfit = GaussNet().fit(X_gaussian, y_gaussian)
_, cvpath = cvfit.cross_validation_path(X_gaussian, y_gaussian, cv=5)
```

```{code-cell} ipython3
ax = cvpath.plot(score='Mean Squared Error');
ax.set_title('Cross-validation Results');
```

The cross-validation curve shows the mean squared error (red dotted line) along with upper and lower standard deviation curves. Two special values are indicated:

- `lambda.min`: the value of $\lambda$ that gives minimum mean cross-validated error
- `lambda.1se`: the value of $\lambda$ that gives the most regularized model such that the cross-validated error is within one standard error of the minimum

We can get the model coefficients at these optimal values:

```{code-cell} ipython3
# Get coefficients at lambda.min
lambda_min = cvpath.index_best['Mean Squared Error']
coef_min, intercept_min = cvfit.interpolate_coefs(lambda_min)
print("Coefficients at lambda.min:")
print(coef_min)
```

## Commonly used function arguments

`GaussNet` provides various arguments for users to customize the fit:

- `alpha`: the elastic net mixing parameter $\alpha$, with range $\alpha \in [0,1]$. $\alpha = 1$ is lasso regression (default) and $\alpha = 0$ is ridge regression.
- `sample_weight`: observation weights, default is 1 for each observation.
- `nlambda`: the number of $\lambda$ values in the sequence (default is 100).
- `lambda_values`: can be provided if the user wants to specify the lambda sequence.
- `standardize`: logical flag for $x$ variable standardization prior to fitting the model sequence.

As an example, we set $\alpha = 0.2$ (more like a ridge regression), and give double weight to the latter half of the observations:

**for weights, use the weight_id!!!!**

```{code-cell} ipython3
# Create weights: double weight for latter half
weights = np.ones(len(X_gaussian))
weights[len(X_gaussian)//2:] = 2

# Fit with alpha=0.2 and custom weights
fit_ridge = GaussNet(alpha=0.2, nlambda=20).fit(X_gaussian, y_gaussian, sample_weight=weights)
fit_ridge.coef_path_.plot();
plt.title('Ridge Regression (α=0.2) with Custom Weights');
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

For example, the following code gives the fitted values for the first 5 observations at $\lambda = 0.05$:

```{code-cell} ipython3
predictions = fit.predict(X_gaussian[:5], interpolation_grid=0.05)
print("Fitted values for first 5 observations at λ = 0.05:")
print(predictions)
```

## Other function arguments

In this section we briefly describe some other useful arguments when calling `GaussNet`: `upper_limits`, `lower_limits`, `penalty_factor`, and `fit_intercept`.

Suppose we want to fit our model but limit the coefficients to be bigger than -0.1 and less than 1.2:

```{code-cell} ipython3
# Fit with coefficient limits
fit_limited = GaussNet(lower_limits=-0.1, upper_limits=1.2).fit(X_gaussian, y_gaussian)
fit_limited.coef_path_.plot();
plt.title('Coefficient Paths with Limits [-0.1, 1.2]');
```

Often we want the coefficients to be positive: to do so, we just need to specify `lower_limits = 0`:

```{code-cell} ipython3
# Fit with non-negative coefficients
fit_positive = GaussNet(lower_limits=0).fit(X_gaussian, y_gaussian)
fit_positive.coef_path_.plot();
plt.title('Non-negative Coefficient Paths');
```

The `penalty_factor` argument allows users to apply separate penalty factors to each coefficient. This is very useful when we have prior knowledge or preference over the variables:

```{code-cell} ipython3
# Set penalty factors for variables 1, 3, and 5 to be zero (no penalty)
penalty_factor = np.ones(X_gaussian.shape[1])
penalty_factor[[0, 2, 4]] = 0  # Variables 1, 3, 5 (0-indexed)

fit_penalty = GaussNet(penalty_factor=penalty_factor).fit(X_gaussian, y_gaussian)
fit_penalty.coef_path_.plot();
plt.title('Coefficient Paths with Custom Penalty Factors');
```

We see from the plot that the three variables with zero penalty factors always stay in the model, while the others follow typical regularization paths and are shrunk to zero eventually.

The `fit_intercept` argument allows the user to decide if an intercept should be included in the model or not (it is never penalized). The default is `fit_intercept = True`:

```{code-cell} ipython3
# Fit without intercept
fit_no_intercept = GaussNet(fit_intercept=False).fit(X_gaussian, y_gaussian)

# Compare with intercept
fit_with_intercept = GaussNet(fit_intercept=True).fit(X_gaussian, y_gaussian)

print("Intercept values:")
print(f"With intercept: {fit_with_intercept.intercepts_[0]:.4f}")
print(f"Without intercept: {fit_no_intercept.intercepts_[0]:.4f}")
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
np.random.seed(42)
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
mfit.coef_path_.plot();
plt.title('Multi-response Linear Regression Coefficient Paths');
```

We can extract the coefficients and make predictions at requested values of $\lambda$:

```{code-cell} ipython3
# Make predictions at specific lambda values
lambda_vals = [0.1, 0.01]
predictions = mfit.predict(X[:5], interpolation_grid=lambda_vals)
print(f"Prediction shape: {predictions.shape}")
```

# Logistic Regression: LogNet

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate binary classification data for testing
X_binomial, y_binomial = make_classification(n_samples=200, n_features=20, n_informative=5, 
                           n_redundant=5, n_clusters_per_class=1, 
                           flip_y=0.3,
                           random_state=42)

# Use the existing binomial data for testing
X_train, X_test, y_train, y_test = train_test_split(X_binomial, y_binomial, test_size=0.5, random_state=42)

fit = LogNet().fit(X_train, y_train)
```

As before, we can print and plot the fitted object, extract the coefficients at specific $\lambda$'s and also make predictions:

```{code-cell} ipython3
fit.coef_path_.plot();
plt.title('Logistic Regression Coefficient Paths');
```

Prediction is a little different for logistic regression, mainly in the function argument `prediction_type`:

- "link" gives the linear predictors
- "response" gives the fitted probabilities
- "class" produces the class label corresponding to the maximum probability

```{code-cell} ipython3
# Make predictions of different types

# Linear predictors
link_pred = fit.predict(X_test[:5], interpolation_grid=0.05, prediction_type="link")
print("Linear predictors (link):")
print(link_pred)

# Probabilities
prob_pred = fit.predict(X_test[:5], interpolation_grid=0.05, prediction_type="response")
print("\nFitted probabilities (response):")
print(prob_pred)

# Class predictions
class_pred = fit.predict(X_test[:5], interpolation_grid=0.05, prediction_type="class")
print("\nClass predictions:")
print(class_pred)
```

For logistic regression, cross-validation has similar arguments and usage as Gaussian:

```{code-cell} ipython3
# Perform cross-validation
cvfit = LogNet().fit(X_binomial, y_binomial)
_, cvpath = cvfit.cross_validation_path(X_binomial, y_binomial, cv=5)
ax = cvpath.plot(score='Binomial Deviance');
ax.set_title('Cross-validation Results for Logistic Regression');
```

# Multinomial Regression: MultiClassNet

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
from sklearn.datasets import make_classification

# Generate multinomial data
X_multinomial, y_multinomial = make_classification(n_samples=100, n_features=20, n_informative=5, 
                          n_redundant=5, n_classes=3, n_clusters_per_class=1, 
                          flip_y=0.2, random_state=42)

# Fit multinomial regression with grouped lasso penalty
fit = MultiClassNet().fit(X_multinomial, y_multinomial)
```

For the `plot` method, we can produce a figure showing the $\ell_2$-norm in one figure:

```{code-cell} ipython3
ax = fit.coef_path_.plot();
ax.set_title('Multinomial Regression Coefficient Paths');
```

We can also do cross-validation:

```{code-cell} ipython3
# Perform cross-validation
cvfit = MultiClassNet().fit(X_multinomial, y_multinomial)
_, cvpath = cvfit.cross_validation_path(X_multinomial, y_multinomial, cv=10);
```

```{code-cell} ipython3
print("Available scores:", list(cvpath.scores.columns))
cvpath.plot(score='Multinomial Deviance');
ax.set_title('Cross-validation Results for Multinomial Regression');
```

Users may wish to predict at the optimally selected $\lambda$:

```{code-cell} ipython3
predictions = fit.predict(X_multinomial[:10], 
                          interpolation_grid=cvpath.index_best['Multinomial Deviance'], 
                          prediction_type="class")
print("Class predictions at lambda.min:")
print(predictions)
```

# Poisson Regression: FishNet

Poisson regression is used to model count data under the assumption of Poisson error, or otherwise non-negative data where the mean and variance are proportional. Like the Gaussian and binomial models, the Poisson distribution is a member of the exponential family of distributions. We usually model its positive mean on the log scale: $\log \mu(x) = \beta_0+\beta' x$.

The log-likelihood for observations $\{x_i,y_i\}_1^N$ is given by
$$
l(\beta|X, Y) = \sum_{i=1}^N \left(y_i (\beta_0+\beta^T x_i) - e^{\beta_0+\beta^Tx_i}\right).
$$
As before, we optimize the penalized log-likelihood:
$$
\min_{\beta_0,\beta} -\frac1N l(\beta|X, Y)  + \lambda \left((1-\alpha) \sum_{i=1}^N \beta_i^2/2 +\alpha \sum_{i=1}^N |\beta_i|\right).
$$

We generate Poisson data:

```{code-cell} ipython3
import numpy as np
from glmnet import FishNet
import matplotlib.pyplot as plt

# Generate Poisson data
np.random.seed(42)
n, p = 100, 20
X_poisson = np.random.randn(n, p)
beta = np.zeros(p)
beta[:5] = [0.5, 0.3, 0.2, 0, 0.8]
log_means = X_poisson @ beta
means = np.exp(log_means)
y_poisson = np.random.poisson(means)

# Fit Poisson regression
fit = FishNet().fit(X_poisson, y_poisson)
```

We plot the coefficients to have a first sense of the result:

```{code-cell} ipython3
fit.coef_path_.plot();
plt.title('Poisson Regression Coefficient Paths');
```

As before, we can extract the coefficients and make predictions at certain $\lambda$'s:

```{code-cell} ipython3
fit.interpolate_coefs(interpolation_grid=1)
predictions = fit.predict(X_poisson[:5], interpolation_grid=[1, 0.1], prediction_type="response")
```

We may also use cross-validation to find the optimal $\lambda$'s:

```{code-cell} ipython3
# Perform cross-validation
cvfit = FishNet().fit(X_poisson, y_poisson)
_, cvpath = cvfit.cross_validation_path(X_poisson, y_poisson, cv=10)
cvpath.plot(score='Poisson Deviance');
plt.title('Cross-validation Results for Poisson Regression');
```

# Cox Regression: CoxNet

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. We have a separate vignette dedicated solely to fitting regularized Cox models with the `glmnet` package; please consult that vignette for details.

# Assessing models on test data

Once we have fit a series of models using `glmnet`, we often assess their performance on a set of evaluation or test data. We usually go through the process of building a prediction matrix, deciding on the performance measure, and computing these measures for a series of values for `lambda`.

## Performance measures

We can compute performance measures on a validation or test dataset. Here's an example for logistic regression:

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate binary classification data for testing
X_test, y_test = make_classification(n_samples=200, n_features=20, n_informative=5, 
                           n_redundant=5, n_clusters_per_class=1, 
                           flip_y=0.3,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

fit = LogNet().fit(X_train, y_train)
```

```{code-cell} ipython3
score_path = fit.score_path(X_test, y_test)
ax = score_path.plot(score='Binomial Deviance');
ax.set_title('Test Set Performance');
```

## ROC curves for binomial data

In the special case of binomial models, users often would like to see the ROC curve for validation or test data:

```{code-cell} ipython3
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve for the best lambda
best_lambda = score_path.index_best['Binomial Deviance']
pred_probs = fit.predict(X_test, interpolation_grid=best_lambda,
                        prediction_type="response")

fpr, tpr, _ = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
```

# Filtering variables

Sometimes we want to filter variables before fitting the model. This can be done using the `exclude` argument:

```{code-cell} ipython3
# Exclude variables 1, 3, and 5 from the model
exclude_vars = [0,1,2] # 0-indexed
fit_excluded = GaussNet(exclude=exclude_vars).fit(X_gaussian, y_gaussian)
fit_excluded.coef_path_.plot()
np.allclose(fit_excluded.coefs_[:,exclude_vars], 0)
```

# Other Package Features

## Offsets

Like other generalized linear models, `glmnet` allows for an "offset". This is a fixed vector of $N$ numbers that is added into the linear predictor. For example, you may have fitted some other logistic regression using other variables (and data), and now you want to see if the present variables can add further predictive power. To do this, you can use the predicted logit from the other model as an offset in the `glmnet` call.

```{code-cell} ipython3
# Example with offset in logistic regression
# Generate offset (e.g., from another model)
offset = np.random.randn(len(X_gaussian))

# Fit with offset
fit_with_offset = LogNet().fit(X_gaussian, y_gaussian, offset=offset)

# Compare with fit without offset
fit_no_offset = LogNet().fit(X_gaussian, y_gaussian)

print("Intercept with offset:", fit_with_offset.intercepts_[0])
print("Intercept without offset:", fit_no_offset.intercepts_[0])
```

## Parallel computing

For large-scale problems, parallel computing can significantly speed up the computation process. The Python `glmnet` package supports parallel processing through joblib:

```{code-cell} ipython3
# Example of parallel cross-validation
from joblib import parallel_backend

# This would be used in practice for large datasets
# with parallel_backend('threading', n_jobs=4):
#     cv_results = fit.cross_validation_path(X, y, cv=10, n_jobs=4)
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

# Multi-response Linear Regression: MultiGaussNet
