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
from glmnet import MultiGaussNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython


n, p, q, nlambda = 103, 17, 3, 100
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
O = rng.standard_normal((n, q)) * 0.2
W = rng.integers(2, 6, size=n)
W[:20] = 3
Y = rng.standard_normal((n, q)) * 5
D = np.column_stack([Y, O, W])
response_id = [f'response[{i}]' for i in range(q)]
offset_id = [f'offset[{i}]' for i in range(q)]
Df = pd.DataFrame(D, columns=response_id + offset_id + ['weight'])
Df

# # Use one of the C++ paths
#
# ## Without CV

GN2 = MultiGaussNet(response_id=response_id,
                    offset_id=offset_id,
                    nlambda=nlambda,
             ).fit(X, Df)

# + magic_args="-i X,Y,O,W,nlambda -o L,C1,C2,C3" language="R"
# library(glmnet)
# GN2 = glmnet(X, Y, 
#              #weights=W, 
#              offset=O,
#              family='mgaussian',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$y1)
# C2 = as.matrix(C$y2)
# C3 = as.matrix(C$y3)
# L = GN2$lambda
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

GN2 = MultiGaussNet(response_id=response_id,
             weight_id='weight',
                    nlambda=nlambda
             )
GN2.fit(X, Df)

# + magic_args="-o L,C1,C2,C3" language="R"
# GN2 = glmnet(X, Y, weights=W, 
#              family='mgaussian',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$y1)
# C2 = as.matrix(C$y2)
# C3 = as.matrix(C$y3)
# L = GN2$lambda
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

GN2 = MultiGaussNet(response_id=response_id,
             offset_id=offset_id,
                    weight_id='weight',
             ).fit(X, Df)


# + magic_args="-o L,C1,C2,C3" language="R"
# GN2 = glmnet(X, 
#              Y, 
#              weights=W, 
#              offset=O,
#              family='mgaussian',
#              nlambda=nlambda)
# C = coef(GN2)
# C1 = as.matrix(C$y1)
# C2 = as.matrix(C$y2)
# C3 = as.matrix(C$y3)
# L = GN2$lambda
#
# -

C = np.array([C1,C2,C3]).T
assert np.allclose(C[:,1:], GN2.coefs_)
assert np.allclose(C[:,0], GN2.intercepts_)

# ## Now with CV, first no weights

GN3 = MultiGaussNet(response_id=response_id,
                    offset_id=offset_id,
                   ).fit(X, Df)


# Capture the fold ids

cv = KFold(5, random_state=0, shuffle=True)
foldid = np.empty(n)
for i, (train, test) in enumerate(cv.split(np.arange(n))):
    foldid[test] = i+1

# ## With an offset using `fraction`

predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Mean Squared Error', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family="mgaussian",
# 		        alignment="fraction",
#                 nlambda=nlambda,
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Mean Squared Error'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'], CVSD) 

GN3.cv_scores_['Mean Squared Error']*q, CVM

# ## With an offset using `lambda`

GN3 = MultiGaussNet(response_id=response_id,
             offset_id=offset_id,
            ).fit(X, Df)
predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN3.plot_cross_validation(score='Mean Squared Error', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 foldid=foldid,
#                 family='mgaussian',
#                 nlambda=nlambda,
# 		alignment="lambda",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Mean Squared Error'], CVM)
assert np.allclose(GN3.cv_scores_['SD(Mean Squared Error)'], CVSD) 

# ## With an offset and weight using `fraction`

GN4 = MultiGaussNet(response_id=response_id,
             offset_id=offset_id,
	         weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction');

GN4.plot_cross_validation(score='Mean Squared Error', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
# 		        alignment="fraction",
#                 family='mgaussian',
#                 nlambda=nlambda,
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Mean Squared Error'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'], CVSD) 

# ## With an offset and weight using `lambda`

GN4 = MultiGaussNet(response_id=response_id,
               offset_id=offset_id,
               weight_id='weight',
               ).fit(X, Df)
predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda');

GN4.plot_cross_validation(score='Mean Squared Error', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# W = as.numeric(W)
# GCV = cv.glmnet(X,
#                 Y, 
#                 offset=O,
#                 weights=W,
#                 foldid=foldid,
#         		alignment="lambda",
#                 family='mgaussian',
#         		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Mean Squared Error'], CVM)
assert np.allclose(GN4.cv_scores_['SD(Mean Squared Error)'], CVSD) 
#
#


