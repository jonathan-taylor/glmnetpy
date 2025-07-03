# GLMNet

The `GLMNet` class is the base estimator for generalized linear models with elastic net regularization. It is not typically used directly, but forms the foundation for specialized classes like `GaussNet`, `LogNet`, `FishNet`, and `CoxNet`.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.glmnet import GLMNet

X, y, coef, intercept = make_dataset(GLMNet, n_samples=100, n_features=10, snr=5)
model = GLMNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.glmnet.GLMNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 