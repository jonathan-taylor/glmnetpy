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

# +
import numpy as np
import pandas as pd
from glmnet.glm import BinomialGLM
from glmnet import GLM, GLMNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython

from glmnet.glm import GLMControl
from glmnet.glmnet import GLMNetControl
control = GLMControl(epsnr=1e-8)
GNcontrol = GLMNetControl()
# -

GNcontrol

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

probit = sm.families.Binomial(link=sm.families.links.Probit())
G1 = GLM(response_id=0,
         offset_id=1,
         weight_id=2,
         family=probit)
G1.fit(X, D)

# ## Also using `pd.DataFrame`

G2 = BinomialGLM(response_id='response',
                 offset_id='offset',
                weight_id='weight', family=probit, summarize=True
                )
G2.fit(X, Df)
assert np.allclose(G2.coef_, G1.coef_)
assert np.allclose(G2.intercept_, G1.intercept_)
G2.summary_
#
#

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# notY = as.integer(1-Y)
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weight=W, 
#         offset=O, 
#         family=binomial(link=probit))
# print(summary(M))
# C = coef(M)
# -

assert np.allclose(G2.coef_, C[1:], rtol=1e-4, atol=1e-4)
assert np.allclose(G2.intercept_, C[0], rtol=1e-4, atol=1e-4)

G2.coef_, C[1:]

# ## Try out dropping weights or offset

G4 = GLM(response_id='binary', family=probit, control=control)
G4.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# Y = as.integer(Y)
# notY = as.integer(1-Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         family=binomial(link=probit),
#        control=glm.control(epsilon=1e-10))
# C = coef(M)
# summary(M)
# -

pred_py = G4.predict(X, prediction_type='link')
pred_R = X @ C[1:] + C[0]
np.linalg.norm(pred_py - pred_R) / np.linalg.norm(pred_py)

(probit.deviance(Df['binary'], probit.link.inverse(pred_py)),
probit.deviance(Df['binary'], probit.link.inverse(pred_R)))

assert np.allclose(G4.coef_, C[1:], rtol=1e-5, atol=1e-5)
assert np.allclose(G4.intercept_, C[0], rtol=1e-5, atol=1e-5)

G5 = GLM(response_id='binary', weight_id='weight', family=probit, control=control)
G5.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         weights=W,
#         family=binomial(link=probit),
#         control=glm.control(epsilon=1e-10))
# C = coef(M)
# -

C[1:], G5.coef_

assert np.allclose(G5.coef_, C[1:], rtol=1e-4, atol=1e-4)
assert np.allclose(G5.intercept_, C[0], rtol=1e-4, atol=1e-4)

pred_py = G5.predict(X, prediction_type='link')
pred_R = X @ C[1:] + C[0]
np.linalg.norm(pred_py - pred_R) / np.linalg.norm(pred_py)

(probit.deviance(Df['binary'], probit.link.inverse(pred_py),
                freq_weights=Df['weight']),
probit.deviance(Df['binary'], probit.link.inverse(pred_R),
                freq_weights=Df['weight']))

G6 = GLM(response_id='binary', offset_id='offset', family=probit, control=control)
G6.fit(X, Df)

# + magic_args="-i R,X,O,W,Y -o C" language="R"
# notY = as.integer(1-Y)
# Y = as.integer(Y)
# W = as.numeric(W)
# M = glm(Y ~ X,
#         offset=O,
#         family=binomial(link=probit),
#        control=glm.control(epsilon=1e-10))
# C = coef(M)
# -

assert np.allclose(G6.coef_, C[1:], rtol=1e-4, atol=1e-4)
assert np.allclose(G6.intercept_, C[0], rtol=1e-4, atol=1e-4)

pred_py = G6.predict(X, prediction_type='link') + Df['offset']
pred_R = X @ C[1:] + C[0] + Df['offset']
np.linalg.norm(pred_py - pred_R) / np.linalg.norm(pred_py)

(probit.deviance(Df['binary'], probit.link.inverse(pred_py)),
probit.deviance(Df['binary'], probit.link.inverse(pred_R)))

# # Try GLMNet (family version)
#
# ## Without CV

GN = GLMNet(response_id='binary',
             offset_id='offset',
             weight_id='weight',
             family=probit,
             control=GNcontrol,
             ).fit(X, Df)


# + magic_args="-o C,L" language="R"
# library(glmnet)
# GN = glmnet(X, Y, 
#             offset=O, 
#             weights=W, 
#             family=binomial(link=probit))
# C = as.matrix(coef(GN))
# L = GN$lambda

# + language="R"
# glm(Y ~ 1, offset=O, weights=W, family=binomial(link=probit))$dev
# -

L.shape, GN.lambda_values_.shape

GN.lambda_max_

assert np.allclose(C.T[10][1:], GN.coefs_[10], rtol=1e-4, atol=1e-4)

GN2 = GLMNet(response_id='binary',
             offset_id='offset',
             weight_id='weight',
             family=probit,
             control=GNcontrol,
             ).fit(X, Df)


# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, 
#              weights=W, 
#              offset=O,
#              family=binomial(link=probit))
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# -

C.T[10][1:]

GN2.coefs_[10]

assert np.allclose(C.T[:50,1:], GN2.coefs_[:50], rtol=1e-3, atol=1e-3)
assert np.allclose(C.T[:50,0], GN2.intercepts_[:50], rtol=1e-3, atol=1e-3)

GN.lambda_values_.shape, L.shape

pred_py = GN2.predict(X, prediction_type='link')[:,10] + Df['offset']
pred_R = X @ C.T[10][1:] + C.T[10][0] + Df['offset']
np.linalg.norm(pred_py - pred_R) / np.linalg.norm(pred_py)

assert np.allclose(L[:50], GN2.lambda_values_[:50])

(probit.deviance(Df['binary'], probit.link.inverse(pred_py),
                freq_weights=Df['weight']) + GN2.lambda_values_[10] * np.fabs(GN.coefs_[10]).sum(),
probit.deviance(Df['binary'], probit.link.inverse(pred_R),
                freq_weights=Df['weight']) + GN2.lambda_values_[10] * np.fabs(C.T[10][1:]).sum())

# ## Now with CV, first no weights

GN3 = GLMNet(response_id='binary', # should be able to use 'response'
               offset_id='offset', 
               family=probit,
               control=GNcontrol
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
#                 family=binomial(link=probit),
# 		alignment="fraction",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50], rtol=1e-3, atol=1e-3)
assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50], rtol=1e-3, atol=1e-3) 

fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50])
ax.axline((CVM.min(),CVM.min()),(CVM.max(),CVM.max()), ls='--', c='k')


fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50])
ax.axline((CVSD.min(),CVSD.min()),(CVSD.max(),CVSD.max()), ls='--', c='k')


# ## With an offset using `lambda`

GN3 = GLMNet(response_id='binary',
             offset_id='offset',
             family=probit,
             control=GNcontrol
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
#                 family=binomial(link=probit),
# 		alignment="lambda",
# 		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50], rtol=1e-3, atol=1e-3)
assert np.allclose(GN3.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50], rtol=1e-3, atol=1e-3) 

# +
fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50])
ax.axline((CVM.min(),CVM.min()),(CVM.max(),CVM.max()), ls='--', c='k')

fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50])
ax.axline((CVSD.min(),CVSD.min()),(CVSD.max(),CVSD.max()), ls='--', c='k')

# -

# ## With an offset and weight using `fraction`

GN4 = GLMNet(response_id='binary',
             offset_id='offset',
	         weight_id='weight',
             family=probit,
             control=GNcontrol,
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
#                 family=binomial(link=probit),
# 		        grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50], rtol=1e-3, atol=1e-3)
assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50], rtol=1e-3, atol=1e-3) 

# +
fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50])
ax.axline((CVM.min(),CVM.min()),(CVM.max(),CVM.max()), ls='--', c='k')

fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50])
ax.axline((CVSD.min(),CVSD.min()),(CVSD.max(),CVSD.max()), ls='--', c='k')

# -

# ## With an offset and weight using `lambda`

GN4 = GLMNet(response_id='binary',
               offset_id='offset',
               weight_id='weight',
             family=probit,
             control=GNcontrol,
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
#                 family=binomial(link=probit),
#         		grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN4.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50], rtol=1e-3, atol=1e-3)
assert np.allclose(GN4.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50], rtol=1e-3, atol=1e-3) 
#
#

# +
fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['Binomial Deviance'].iloc[:50], CVM[:50])
ax.axline((CVM.min(),CVM.min()),(CVM.max(),CVM.max()), ls='--', c='k')

fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['SD(Binomial Deviance)'].iloc[:50], CVSD[:50])
ax.axline((CVSD.min(),CVSD.min()),(CVSD.max(),CVSD.max()), ls='--', c='k')

# -

GN2.summary_

# + magic_args="-o C,L" language="R"
# GN2 = glmnet(X, Y, 
#              weights=W, 
#              offset=O,
#              family=binomial(link=probit))
# C = as.matrix(coef(GN2))
# L = GN2$lambda
# GN2
# -


