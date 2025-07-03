# MultiGaussNet

The `MultiGaussNet` class fits a multi-response linear regression model with elastic net regularization (Gaussian family). It is suitable for continuous response data with multiple outputs.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.multigaussnet import MultiGaussNet

X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=100, n_features=10, 
                                    n_targets=3, snr=5)
model = MultiGaussNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.multigaussnet.MultiGaussNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 