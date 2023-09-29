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
n, p, q = 200, 8, 3
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))

from glmnet import MultiGaussNet
nlambda = 7
W = np.ones(n)
W[:10] = 0
L = MultiGaussNet(nlambda=nlambda)
L.fit(X, Y, sample_weight=W)


import rpy2
# %load_ext rpy2.ipython
# %R -i X,Y,nlambda,W

X[0]

# + language="R"
# library(glmnet)
# G = glmnet(X, Y, family='mgaussian', nlambda=nlambda, weights=W)
# plot(G)
# G

# +
#ax = L.plot_coefficients(xvar='norm')
# -

L.summary_

# + magic_args="-o C_R1,C_R2,C_R3,I_R1,I_R2,I_R3" language="R"
# C_R1 = as.matrix(coef(G)$y1)[-1,]
# C_R2 = as.matrix(coef(G)$y2)[-1,]
# C_R3 = as.matrix(coef(G)$y3)[-1,]
# I_R1 = as.matrix(coef(G)$y1)[1,]
# I_R2 = as.matrix(coef(G)$y2)[1,]
# I_R3 = as.matrix(coef(G)$y3)[1,]
#
# -

C_R = np.concatenate([C_R1.T[:,:,None], C_R2.T[:,:,None], C_R3.T[:,:,None]], axis=2)
C_R.shape

np.linalg.norm(C_R - L.coefs_) / np.linalg.norm(L.coefs_)

I_R = np.array([I_R1,
                I_R2,
                I_R3])
np.linalg.norm(I_R.T - L.intercepts_) / np.linalg.norm(L.intercepts_)

L.intercepts_.shape, L.coefs_.shape

I_R.T


