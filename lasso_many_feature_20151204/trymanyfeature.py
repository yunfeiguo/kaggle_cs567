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
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

parser = argparse.ArgumentParser()
parser.add_argument('-f',nargs='+',required=True)
args = parser.parse_args()


#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggsum_cleaned_smalltrain.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggsum_cleaned_train.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

def runModel(i,featureCombo):
    mae = np.array([])   
    logging.warning('try alpha = %s' % i)
    coef = None
    for ktrain,ktest in kf:
        #x = trainCleaned.iloc[ktrain,]
        #y = trainCleaned.iloc[ktest,]    
	#x = x[x.Expected < 10]
        #model = linear_model.Lasso(alpha = i)
        #model.fit(x[featureCombo],x['Expected'])
        #mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
	predictor = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,featureCombo)]
	predictor2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,featureCombo)]
	out = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'Expected')]
	out2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'Expected')]
	predictor = predictor[out.Expected < 10]
	out = out[out.Expected < 10]
        model = linear_model.Lasso(alpha = i)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('coef: %s' % coef)
    logging.warning('average 50-fold MAE for alpha %s feature %s: %s ' % (i,featureCombo,mae.mean()))

#add interactions
cols = trainCleaned.columns
for i in cols:
    for j in cols:
        trainCleaned[str(i+'X'+j)] = trainCleaned[i]*trainCleaned[j]
#add square terms
#add log terms
#add inverse
#add 

kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
#for featureCombo in [['RhoHV'],['RhoHV','Ref'],['RhoHV','Zdr'],['Ref','Kdp'],['Zdr','radardist_km'],['Kdp','radardist_km'],['KdpXradardist_km','Kdp','radardist_km'],['Kdp','Zdr','radardist_km']]:	
#for featureCombo in [args.f]:	
featureCombo = trainCleaned.columns
logging.warning('try feature %s' % featureCombo)
allMAE = []
jobs = []
#for i in [0.001,0.1,0.5,1,2,4,8,12,20]:
#for i in [1e-10,1e-8,1e-6,1e-4,1e-3]:
for i in args.f:
    runModel(float(i),featureCombo)
