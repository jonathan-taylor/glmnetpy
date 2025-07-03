---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# LogNet

The `LogNet` class fits a logistic regression model with elastic net regularization (binomial family). It is suitable for binary classification problems.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.lognet import LogNet

X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
model = LogNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.lognet.LogNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 