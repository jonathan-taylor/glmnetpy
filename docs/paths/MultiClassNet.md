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

# MultiClassNet

The `MultiClassNet` class fits a multinomial logistic regression model with elastic net regularization (multinomial family). It is suitable for multi-class classification problems.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.multiclassnet import MultiClassNet

X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=100, n_features=10, 
                                    n_classes=3, snr=5)
model = MultiClassNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.multiclassnet.MultiClassNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 