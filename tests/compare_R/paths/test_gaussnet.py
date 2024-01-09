# ---
# jupyter:
#   jupytext:
#     formats: ipynb,Rmd,py
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
import pandas as pd
from glmnet import GaussNet, GLM, GLMNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython


n, p = 103, 20
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
O = rng.standard_normal(n) * 0.2
W = rng.integers(2, 6, size=n)
W[:20] = 3
Y = rng.standard_normal(n) * 5
D = np.array([Y, O, W]).T
Df = pd.DataFrame(D, columns=['response', 'offset', 'weight'])
Df

# # Check a GLM

G1 = GLM(response_id=0,
         offset_id=1,
         weight_id=2,
        family=sm.families.Gaussian(), summarize=True)
G1.fit(X, D)
G1.summary_
G1.dispersion_

# ## Also using `pd.DataFrame`

G2 = GLM(response_id='response',
         offset_id='offset',
         weight_id='weight',
         family=sm.families.Gaussian(),
         summarize=True
                )
G2.fit(X, Df)
assert np.allclose(G2.coef_, G1.coef_)
assert np.allclose(G2.intercept_, G1.intercept_)
print(G2.summary_)
#
#

# + magic_args="-i X,O,W,Y -o C" language="R"
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weight=W, 
#         offset=O)
# C = coef(M)
# summary(M)
# -

assert np.allclose(G2.coef_, C[1:])
assert np.allclose(G2.intercept_, C[0])

C[1:], G2.coef_

# ## Try out dropping weights or offset

G4 = GLM(response_id='response',
         family=sm.families.Gaussian())
G4.fit(X, Df)

# + magic_args="-i X,O,W,Y -o C" language="R"
#
# W = as.numeric(W)
# M = glm(Y ~ X,
#         family=gaussian)
# C = coef(M)
# -

assert np.allclose(G4.coef_, C[1:], rtol=1e-5, atol=1e-5)
assert np.allclose(G4.intercept_, C[0], rtol=1e-5, atol=1e-5)

G5 = GLM(response_id='response', weight_id='weight', family=sm.families.Gaussian())
G5.fit(X, Df)

# + magic_args="-i X,O,W,Y -o C" language="R"
#
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weights=W,
#         family=gaussian)
# C = coef(M)
# -

assert np.allclose(G5.coef_, C[1:])
assert np.allclose(G5.intercept_, C[0])

G6 = GLM(response_id='response', offset_id='offset', family=sm.families.Gaussian())
G6.fit(X, Df)

# + magic_args="-i X,O,W,Y -o C" language="R"
#
# W = as.numeric(W)
# M = glm(Y ~ X,
#         offset=O,
#         family=gaussian)
# C = coef(M)
# -

assert np.allclose(G6.coef_, C[1:])
assert np.allclose(G6.intercept_, C[0])

# # Try GLMNet (family version)

GN = GLMNet(response_id='response',
            offset_id='offset',
            weight_id='weight',
           family=sm.families.Gaussian())
GN.fit(X, Df)

GN.null_deviance_

# + magic_args="-o C,L" language="R"
# library(glmnet)
# GN = glmnet(X, Y, offset=O, weights=W, family='gaussian')
# C = as.matrix(coef(GN))
# L = GN$lambda
# -

assert np.allclose(C.T[30][1:], GN.coefs_[30], rtol=1e-3, atol=1e-3)


GN.coefs_[30]

C.T[30][1:]

# # Use one of the C++ paths
#
# ## Without CV

GN2 = GaussNet(response_id='response',
              offset_id='offset',
              weight_id='weight',
             ).fit(X, Df)


GN = GLMNet(response_id='response',
              family=sm.families.Gaussian()
             ).fit(X, Df)
GN.summary_

# + magic_args="-o C,L" language="R"
# GN = glmnet(X, Y,
#              family='gaussian')
# print(GN)

# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, weights=W, 
#              offset=O,
#              family='gaussian')
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# -

assert np.allclose(C.T[:,1:], GN2.coefs_)
assert np.allclose(C[0], GN2.intercepts_)

GN2 = GaussNet(response_id='response',
             weight_id='weight',
             ).fit(X, Df)


# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, weights=W, 
#              family='gaussian')
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# -

assert np.allclose(C.T[:,1:], GN2.coefs_)
assert np.allclose(C[0], GN2.intercepts_)

GN2 = GaussNet(response_id='response',
             offset_id='offset',
             ).fit(X, Df)


# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, 
#              offset=O, 
#              family='gaussian')
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# -

GN2.coefs_[10]

C.T[:,1:][10]

assert np.allclose(C.T[:,1:], GN2.coefs_)
assert np.allclose(C[0], GN2.intercepts_)

# ## Now with CV, first no weights

GN3 = GaussNet(response_id='response',
               offset_id='offset',
               ).fit(X, Df)


# Capture the fold ids

cv = KFold(5, random_state=0, shuffle=True)
foldid = np.empty(n)
for i, (train, test) in enumerate(cv.split(np.arange(n))):
    foldid[test] = i+1

# ## With an offset using `fraction`

predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Gaussian Deviance', xvar='dev')

GN3.summary_

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# Y = as.numeric(Y)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family="gaussian",
# 		alignment="fraction",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Gaussian Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Gaussian Deviance)'], CVSD) 

# ## With an offset using `lambda`

GN3 = GaussNet(response_id='response',
             offset_id='offset',
            ).fit(X, Df)
predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN3.plot_cross_validation(score='Gaussian Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family='gaussian',
# 		alignment="lambda",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Gaussian Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Gaussian Deviance)'], CVSD) 

# ## With an offset and weight using `fraction`

GN4 = GaussNet(response_id='response',
             offset_id='offset',
	         weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction');

np.log(GN4.lambda_values_)

GN4.plot_cross_validation(score='Gaussian Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
# 		        alignment="fraction",
#                 family='gaussian',
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Gaussian Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Gaussian Deviance)'], CVSD) 

# ## With an offset and weight using `lambda`

GN4 = GaussNet(response_id='response',
               offset_id='offset',
               weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN4.plot_cross_validation(score='Gaussian Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
#         		alignment="lambda",
#                 family='gaussian',
#         		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Gaussian Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Gaussian Deviance)'], CVSD) 
#
#


