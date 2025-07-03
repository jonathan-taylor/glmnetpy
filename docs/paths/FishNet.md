# FishNet

The `FishNet` class fits a Poisson regression model with elastic net regularization (Poisson family). It is suitable for count data and event rate modeling.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.fishnet import FishNet

X, y, coef, intercept = make_dataset(FishNet, n_samples=100, n_features=10, snr=5)
model = FishNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.fishnet.FishNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 