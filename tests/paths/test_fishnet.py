# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
rng = np.random.default_rng(0)
n, p = 4, 5
X = rng.standard_normal((n, p))
Y = rng.poisson(lam=20, size=n)

from glmnet import FishNet
nlambda, lambda_min_ratio = 3, 0.2
L = FishNet(nlambda=nlambda, lambda_min_ratio=lambda_min_ratio)
L.fit(X, Y)


import rpy2
# %load_ext rpy2.ipython
# %R -i X,Y,nlambda,lambda_min_ratio

# + language="R"
# library(glmnet)
# G = glmnet(X, Y, family='poisson', nlam=nlambda, lambda.min.ratio=lambda_min_ratio)
# plot(G)
# G
# -

ax = L.plot_coefficients(xvar='norm')

L.summary_

# + magic_args="-o C_R,I_R" language="R"
# C_R = as.matrix(coef(G))[-1,]
# I_R = as.matrix(coef(G))[1,]
# -

np.linalg.norm(C_R.T - L.coefs_) / np.linalg.norm(L.coefs_)

np.linalg.norm(I_R - L.intercepts_) / np.linalg.norm(L.intercepts_)

L.coefs_.shape

L._fit['nin'].max()


