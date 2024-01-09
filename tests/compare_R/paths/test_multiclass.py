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

import string
import numpy as np
import pandas as pd
from glmnet import MultiClassNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython


n, p, q, nlambda = 103, 17, 3, 100
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
O = rng.standard_normal((n, q)) * 0.2
W = rng.integers(2, 6, size=n)
W[:20] = 3
labels = list(string.ascii_uppercase[:q])
Y = rng.choice(labels, size=n)
R = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape((-1,1)))
response_id = ['response']
offset_id = [f'offset[{i}]' for i in range(q)]
Df = pd.DataFrame({'response':Y,
                   'weight':W})
for i, l in enumerate(offset_id):
    Df[l] = O[:,i]
Df

# # Use one of the C++ paths
#
# ## Without CV

GN2 = MultiClassNet(response_id='response',
                    offset_id=offset_id,
                    nlambda=nlambda,
             ).fit(X, Df)

# + magic_args="-i R,X,Y,O,W,nlambda,n,q -o L,C1,C2,C3" language="R"
# library(glmnet)
# colnames(R) <- c('A', 'B', 'C') # will need to change if we change q
# colnames(O) <- c('A', 'B', 'C')
# GN2 = glmnet(X, 
#              R, 
#              #weights=W, 
#              offset=O,
#              family='multinomial',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$A)
# C2 = as.matrix(C$B)
# C3 = as.matrix(C$C)
# L = GN2$lambda
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

GN2 = MultiClassNet(response_id=response_id,
                    weight_id='weight',
                    nlambda=nlambda
             )
GN2.fit(X, Df)

# + magic_args="-o L,C1,C2,C3" language="R"
# W = as.numeric(W)
# GN2 = glmnet(X, R, weights=W, 
#              family='multinomial',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$A)
# C2 = as.matrix(C$B)
# C3 = as.matrix(C$C)
# L = GN2$lambda
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

GN2 = MultiClassNet(response_id=response_id,
                    offset_id=offset_id,
                    weight_id='weight',
                    nlambda=nlambda,
             ).fit(X, Df)


# + magic_args="-o L,C1,C2,C3" language="R"
# GN2 = glmnet(X, 
#              R, 
#              weights=W, 
#              offset=O,
#              family='multinomial',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$A)
# C2 = as.matrix(C$B)
# C3 = as.matrix(C$C)
# L = GN2$lambda
#
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

# ## Now with CV, first no weights

GN3 = MultiClassNet(response_id=response_id,
                    offset_id=offset_id,
                   ).fit(X, Df)


# Capture the fold ids

cv = KFold(5, random_state=0, shuffle=True)
foldid = np.empty(n)
for i, (train, test) in enumerate(cv.split(np.arange(n))):
    foldid[test] = i+1

# ## With an offset using `fraction`

predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Multinomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 R, 
#                 offset=O,
#                 foldid=foldid,
#                 family="multinomial",
# 		        alignment="fraction",
#                 nlambda=nlambda,
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Multinomial Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Multinomial Deviance)'], CVSD) 

# ## With an offset using `lambda`

GN3 = MultiClassNet(response_id=response_id,
             offset_id=offset_id,
            ).fit(X, Df)
predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN3.plot_cross_validation(score='Multinomial Deviance', xvar='lambda')

GN3.plot_cross_validation(score='Accuracy', xvar='lambda')

GN3.plot_cross_validation(score='Misclassification Error', xvar='lambda')

GN3.cv_scores_.columns

# + magic_args="-i foldid -o CVM,CVSD,CVM_A,CVSD_A" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 R, 
#                 offset=O,
#                 foldid=foldid,
#                 family='multinomial',
#                 nlambda=nlambda,
# 		alignment="lambda",
# 		grouped=TRUE)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# GCV = cv.glmnet(X,
#                 R, 
#                 offset=O,
#                 foldid=foldid,
#                 family='multinomial',
#                 nlambda=nlambda,
# 		        alignment="lambda",
#                 type.measure='class',
# 		grouped=TRUE)
# plot(GCV)
# CVM_A = GCV$cvm
# CVSD_A = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Multinomial Deviance'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Multinomial Deviance)'], CVSD) 

assert np.allclose(GN3.cv_scores_['Accuracy'], 1 - CVM_A)
assert np.allclose(GN3.cv_scores_['SD(Accuracy)'], CVSD_A) 

assert np.allclose(GN3.cv_scores_['Misclassification Error'], CVM_A)
assert np.allclose(GN3.cv_scores_['SD(Misclassification Error)'], CVSD_A) 

# ## With an offset and weight using `fraction`

GN4 = MultiClassNet(response_id='response',
             offset_id=offset_id,
	         weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN4.plot_cross_validation(score='Multinomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 R, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
# 		        alignment="fraction",
#                 family='multinomial',
#                 nlambda=nlambda,
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Multinomial Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Multinomial Deviance)'], CVSD) 

# ## With an offset and weight using `lambda`

GN4 = MultiClassNet(response_id=response_id,
               offset_id=offset_id,
               weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN4.plot_cross_validation(score='Multinomial Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 R, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
#         		alignment="lambda",
#                 family='multinomial',
#         		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Multinomial Deviance'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Multinomial Deviance)'], CVSD) 
#
#


