# LogNet

The `LogNet` class fits a logistic regression model with elastic net regularization (binomial family). It is suitable for binary classification problems.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.lognet import LogNet

X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
model = LogNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.lognet.LogNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 