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
from glmnet.glm import BinomialGLM
from glmnet import LogNet, GLM, GLMNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython


n, p = 103, 20
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
R = rng.choice(['D', 'C'], size=n) 
O = rng.standard_normal(n) * 0.2
W = rng.integers(2, 6, size=n)
W[:20] = 3
W = W / W.mean()
L = LabelEncoder().fit(R)
Y = R==L.classes_[1]
D = np.array([Y, O, W]).T
Df = pd.DataFrame(D, columns=['binary', 'offset', 'weight'])
Df['response'] = R
Df

# # Check a GLM

G1 = GLM(response_id=0,
         offset_id=1,
         weight_id=2,
        family=sm.families.Binomial())
G1.fit(X, D)

# ## Also using `pd.DataFrame`

G2 = BinomialGLM(response_id='response',
                 offset_id='offset',
                weight_id='weight'
                )
G2.fit(X, Df)
assert np.allclose(G2.coef_, G1.coef_)
assert np.allclose(G2.intercept_, G1.intercept_)
#
#

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# notY = as.integer(1-Y)
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weight=W, 
#         offset=O, 
#         family=binomial)
# C = coef(M)
# -

assert np.allclose(G2.coef_, C[1:])
assert np.allclose(G2.intercept_, C[0])

# ## Try out dropping weights or offset

G4 = BinomialGLM(response_id='binary')
G4.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# Y = as.integer(Y)
# notY = as.integer(1-Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         family=binomial)
# C = coef(M)
# -

assert np.allclose(G4.coef_, C[1:])
assert np.allclose(G4.intercept_, C[0])

G5 = BinomialGLM(response_id='binary', weight_id='weight')
G5.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weights=W,
#         family=binomial)
# C = coef(M)
# -

C[1:], G5.coef_

assert np.allclose(G5.coef_, C[1:])
assert np.allclose(G5.intercept_, C[0])

G6 = BinomialGLM(response_id='binary', offset_id='offset')
G6.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# notY = as.integer(1-Y)
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         offset=O,
#         family=binomial)
# C = coef(M)
# -

assert np.allclose(G6.coef_, C[1:])
assert np.allclose(G6.intercept_, C[0])

# # Try GLMNet (family version)

GN = GLMNet(response_id='binary',
            offset_id='offset',
            weight_id='weight',
           family=sm.families.Binomial())
GN.fit(X, Df)

# + magic_args="-o C,L" language="R"
# library(glmnet)
# GN = glmnet(X, Y, offset=O, weights=W, family='binomial')
# C = as.matrix(coef(GN))
# L = GN$lambda
# -

assert np.allclose(C.T[10][1:], GN.coefs_[10], rtol=1e-4, atol=1e-4)

# # Use one of the C++ paths
#
# ## Without CV

GN2 = LogNet(response_id='response',
            offset_id='offset',
             weight_id='weight',
             ).fit(X, Df)


# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, weights=W, 
#              offset=O,
#              family='binomial')
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# -

C.T[10][1:]

GN2.coefs_[10]

assert np.allclose(C.T[:,1:], GN2.coefs_)
assert np.allclose(C[0], GN2.intercepts_)

GN2._family

# ## Now with CV, first no weights

GN3 = LogNet(response_id='response',
               offset_id='offset',
               ).fit(X, Df)


# Capture the fold ids

cv = KFold(5, random_state=0, shuffle=True)
foldid = np.empty(n)
for i, (train, test) in enumerate(cv.split(np.arange(n))):
    foldid[test] = i+1

# ## With an offset using `fraction`

predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Binomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# R = as.numeric(R)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family='binomial',
# 		alignment="fraction",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Binomial Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'], CVSD) 

# ## With an offset using `lambda`

GN3 = LogNet(response_id='response',
             offset_id='offset',
            ).fit(X, Df)
predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN3.plot_cross_validation(score='Binomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# R = as.numeric(R)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family='binomial',
# 		alignment="lambda",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Binomial Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'], CVSD) 

# ## With an offset and weight using `fraction`

GN4 = LogNet(response_id='response',
             offset_id='offset',
	         weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN4.plot_cross_validation(score='Binomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# R = as.numeric(R)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
# 		        alignment="fraction",
#                 family='binomial',
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Binomial Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'], CVSD) 

# ## With an offset and weight using `lambda`

GN4 = LogNet(response_id='response',
               offset_id='offset',
               weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN4.plot_cross_validation(score='Binomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# O = as.numeric(O)
# R = as.numeric(R)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
#         		alignment="lambda",
#                 family='binomial',
#         		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Binomial Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'], CVSD) 
#
#


