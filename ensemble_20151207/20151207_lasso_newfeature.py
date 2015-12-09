#!/usr/bin/env python
import os
import sys
sys.path.append("/home/yunfeiguo/projects/kaggle_cs567/src")
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import dask.dataframe as dd
from yg_utils import *
from sklearn import linear_model
import logging
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

#reading in training data
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.v2.csv'
fn_out_train='/home/yunfeiguo/projects/kaggle_cs567/results/20151207_lasso_pred_train.csv'
fn_out_test='/home/yunfeiguo/projects/kaggle_cs567/results/20151207_lasso_pred_test.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

x = trainCleaned
x = x[x['Expected'] < 10]
y = testData
logging.warning('preprocessing done')
logging.warning('start modeling')
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
alpha = 1e-6
xModel = linear_model.Lasso(alpha,max_iter=1000000)
xModel.fit(x[featureCombo],x['Expected'])
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(xModel.coef_))
z = pd.DataFrame(pd.DataFrame.dot(y[featureCombo],xModel.coef_))
#z = xModel.predict(y[featureCombo])
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out_test)
logging.warning('Prediction done')

z = pd.DataFrame(pd.DataFrame.dot(trainCleaned[featureCombo],xModel.coef_))
#z = xModel.predict(y[featureCombo])
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out_train)
logging.warning('Prediction done')
