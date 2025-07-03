# GaussNet

The `GaussNet` class fits a linear regression model with elastic net regularization (Gaussian family). It is suitable for continuous response data.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.gaussnet import GaussNet

X, y, coef, intercept = make_dataset(GaussNet, n_samples=100, n_features=10, snr=5)
model = GaussNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.gaussnet.GaussNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 