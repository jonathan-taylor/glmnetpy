# CoxNet

The `CoxNet` class fits a Cox proportional hazards model with elastic net regularization. It is suitable for survival (time-to-event) analysis with censored data.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_survival
from glmnet.cox import CoxNet

X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
model = CoxNet()
model.fit(X, y)
print(model.coef_)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.cox.CoxNet
    :members:
    :inherited-members:
    :show-inheritance:
``` 