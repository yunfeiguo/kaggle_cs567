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
#parser.add_argument('-l',nargs='+',required=True)
args = parser.parse_args()


#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggsum_cleaned_smalltrain.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

#def runModel(i,j,featureCombo):
def runModel(i,featureCombo):
    mae = np.array([])   
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
        model = linear_model.Lasso(alpha = i,max_iter=10000,warm_start=True)
        #model = linear_model.SGDRegressor(learning_rate='optimal',loss=j,alpha = i,n_iter=50,epsilon=5)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('coef: %s' % coef)
    #logging.warning('loss %s average 50-fold MAE for alpha %s feature %s: %s ' % (j,i,featureCombo,mae.mean()))
    logging.warning('average 50-fold MAE for alpha %s feature %s: %s ' % (i,featureCombo,mae.mean()))

kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
featureCombo = trainCleaned.columns
logging.warning('try feature %s' % featureCombo)
allMAE = []
jobs = []
for i in args.f:
    #for j in args.l:
        #runModel(float(i),j,featureCombo)
    runModel(float(i),featureCombo)
