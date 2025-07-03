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

# GaussNet

The `GaussNet` class fits a linear regression model with elastic net regularization (Gaussian family). It is suitable for continuous response data.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.gaussnet import GaussNet

X, y, coef, intercept = make_dataset(GaussNet, n_samples=100, n_features=10, snr=5)
model = GaussNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.gaussnet.GaussNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 