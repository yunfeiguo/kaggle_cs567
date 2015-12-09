#!/usr/bin/env python
import os
import sys
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
fn_train='data/aggmean_cleaned_smalltrain.csv'
fn_test='data/aggmean_cleaned_smalltest.csv'
fn_out='data/elasticNet_pred.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

x = trainCleaned
y = testData
x['RhoHVXradardist_km'] = x['RhoHV']*x['radardist_km']
y['RhoHVXradardist_km'] = y['RhoHV']*y['radardist_km']
x = x[x['Expected'] < 10]
logging.warning('preprocessing done')
logging.warning('start modeling')
featureCombo = ['RhoHV', 'RhoHVXradardist_km', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
alpha = 1e-6
l1ratio = 0.5
#xModel = linear_model.Lasso(alpha)
xModel = linear_model.ElasticNet(alpha = alpha,l1_ratio=l1ratio,max_iter = 10000)
xModel.fit(x[featureCombo],x['Expected'])
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(xModel.coef_))
z = pd.DataFrame(xModel.predict(y[featureCombo]),index=y.index)
#z = xModel.predict(y[featureCombo])
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out)
logging.warning('Prediction done')
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
