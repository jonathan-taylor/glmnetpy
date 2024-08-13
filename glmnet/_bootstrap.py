import numpy as np
import pandas as pd

from sklearn.base import clone

from .base import Design
from .glm import (compute_grad,
                  GLM)

def bootstrap_GLM(X, 
                  Df,
                  glm=None,
                  active_set=None, 
                  inactive_set=None,
                  rng=None, B=500):
    
    if glm is None:
        glm = GLM()

    n = X.shape[0]
    
    if rng is None:
        rng = np.random.default_rng()

    if active_set is not None:
        if type(X) == np.ndarray:
            X_a = X[:,active_set]
        else:
            X_a = X.loc[:,active_set]
    else:
        X_a = X

    # find a warm start
    
    glm_ = clone(glm)
    warm_state = glm_.fit(X_a, Df).state_
    
    _, Y, R, O, W = glm.get_data_arrays(X, Df)

    prob = W / W.sum()
    D = Design(X,
               W,
               standardize=glm.standardize,
               intercept=glm.fit_intercept)

    coefs = []
    grads = []

    for _ in range(B):

        idx = rng.choice(n, n, replace=True, p=prob)
        W_idx = np.bincount(idx, minlength=X.shape[0])

        glm_ = clone(glm)
        glm_.summarize = True
        glm_.weight_id = f'boot_weight_{id(glm_)}'
        glm_.response_id = f'response_{id(glm_)}'
        Df_star = pd.DataFrame({f'response_{id(glm_)}': R,
                                f'boot_weight_{id(glm_)}': W_idx})
        if O is not None:
            Df_star[f'offset_{id(glm_)}'] = O
            glm_.offset_id = f'offset_{id(glm_)}'

        # previous method, not using weights
        # if type(X_a) == np.ndarray:
        #     X_star = X_a[idx]
        # else:
        #     X_star = X_a.iloc[idx]
        # if type(Df) == np.ndarray:
        #     Df_star = Df[idx]
        # else:
        #     Df_star = Df.iloc[idx]

        glm_.fit(X_a,
                 Df_star,
                 warm_state=warm_state)
        coefs.append(np.hstack([glm_.intercept_, glm_.coef_.copy()]))

        full_coef = np.zeros(X.shape[1])
        if active_set is not None:
            full_coef[active_set] = glm_.coef_
        else:
            full_coef[:] = glm_.coef_
        if inactive_set is not None:
            D_star = Design(X,
                            W_idx,
                            standardize=glm_.standardize,
                            intercept=glm_.fit_intercept)
            grad = compute_grad(glm_, 
                                glm_.intercept_,
                                full_coef,
                                D_star,
                                R,
                                offset=O,
                                sample_weight=W_idx)[0] 
            # drop the coef and keep inactive coordinates
            grads.append(grad[1:][inactive_set]) 
        else:
            grads.append(np.nan)

    return np.array(coefs), np.array(grads)
