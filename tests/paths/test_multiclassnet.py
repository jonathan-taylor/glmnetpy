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
n, p = 200, 4
X = rng.standard_normal((n, p))
Y = rng.choice(['A','B', 'C'], size=n)

from glmnet import MultiClassNet
nlambda = 200
W = np.ones(n)
W[:10] = 0
L = MultiClassNet(nlambda=nlambda, grouped=True)
L.fit(X, Y, sample_weight=W)


import rpy2
# %load_ext rpy2.ipython
# %R -i X,Y,nlambda,W

X[0]

# + language="R"
# library(glmnet)
# y = as.factor(Y)
# X = as.matrix(X)
# W = as.numeric(W)
# G = glmnet(X, y, family='multinomial', nlam=nlambda, weights=W, type.multinomial='grouped')
# #plot(G)
# G
#

# +
#ax = L.plot_coefficients(xvar='norm')
# -

L.summary_

# + magic_args="-o A0" language="R"
# A0 = G$a0

# + magic_args="-o C_R1,C_R2,C_R3,I_R1,I_R2,I_R3" language="R"
# C_R1 = as.matrix(coef(G)$A)[-1,]
# C_R2 = as.matrix(coef(G)$B)[-1,]
# C_R3 = as.matrix(coef(G)$C)[-1,]
# I_R1 = as.matrix(coef(G)$A)[1,]
# I_R2 = as.matrix(coef(G)$B)[1,]
# I_R3 = as.matrix(coef(G)$C)[1,]
#
# -

C_R = np.concatenate([C_R1.T[:,:,None], C_R2.T[:,:,None], C_R3.T[:,:,None]], axis=2)
C_R.shape, L.coefs_.shape

np.linalg.norm(C_R - L.coefs_[:C_R.shape[0]]) / np.linalg.norm(L.coefs_)

I_R = np.array([I_R1,
                I_R2,
                I_R3])
np.linalg.norm(I_R.T - L.intercepts_[:I_R.shape[1]]) / np.linalg.norm(L.intercepts_)

L.coefs_[10]


