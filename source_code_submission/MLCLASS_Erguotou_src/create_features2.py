#!/usr/bin/env python
# This script aims to predict test data with model trained by full train data
# set, then use the predicted result as new features in ensemble training

import os
import numpy as np
import pandas as pd
import random
from yg_utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold

fn_train='data/aggmean_cleaned_smalltrain_reducedFeature.csv'
train=pd.read_csv(fn_train,sep=',',index_col='Id')

fn_test='data/aggmean_cleaned_smalltest.csv'
test=pd.read_csv(fn_test,sep=',',index_col='Id')

classification_featureCombo = train.columns[train.columns!='Expected']

# segregate data to to get train data with expected<100
train['class_exp']=pd.cut(train['Expected'],[0,100,1e6],labels=[0,1])
train1=train.loc[train['class_exp']==0]

# Linear Ridge
from sklearn import linear_model
model=linear_model.Ridge(alpha = 1e-6)
model.fit(train1[classification_featureCombo],train1['Expected'])
pred_tr=model.predict(train[classification_featureCombo])
pred_te=model.predict(test[classification_featureCombo])
train['linearRidge_pred']=pred_tr
test['linearRidge_pred']=pred_te
print('LR finished')

# Bayesian Ridge
from sklearn import linear_model
alpha=1e-6
lamda=1e-6
model=linear_model.BayesianRidge(alpha_1=alpha,alpha_2=alpha,lambda_1=lamda,lambda_2=lamda)
model.fit(train1[classification_featureCombo],train1['Expected'])
pred_tr=model.predict(train[classification_featureCombo])
pred_te=model.predict(test[classification_featureCombo])
train['Bay_pred']=pred_tr
test['Bay_pred']=pred_te
print('Bayesian finished')

# Random Forest
from sklearn.ensemble import RandomForestRegressor
K=500
md=10
model = RandomForestRegressor(n_estimators=K,max_depth=md)
model.fit(train1[classification_featureCombo],train1['Expected'])
pred_tr=model.predict(train[classification_featureCombo])
pred_te=model.predict(test[classification_featureCombo])
train['RF_pred']=pred_tr
test['RF_pred']=pred_te
print('RF finished')

# GBR
from sklearn.ensemble import GradientBoostingRegressor
K=1000
md=5
model=GradientBoostingRegressor(n_estimators=K, learning_rate=0.1,max_depth=md, random_state=0, loss='ls')
model.fit(train1[classification_featureCombo],train1['Expected'])
pred_tr=model.predict(train[classification_featureCombo])
pred_te=model.predict(test[classification_featureCombo])
train['boost_pred']=pred_tr
test['boost_pred']=pred_te
print('GBR finished')

# RANSAC
from sklearn import linear_model
model = linear_model.RANSACRegressor(linear_model.LinearRegression())
model.fit(train1[classification_featureCombo],train1['Expected'])
pred_tr=model.predict(train[classification_featureCombo])
pred_te=model.predict(test[classification_featureCombo])
train['ransac_pred']=pred_tr
test['ransac_pred']=pred_te
print('RANSAC finished')

train.to_csv('data/aggmean_cleaned_train_ModelFeature.csv')
test.to_csv('data/aggmean_cleaned_test_ModelFeature.csv')

