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
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

parser = argparse.ArgumentParser()
parser.add_argument('-f',nargs='+',required=True)
args = parser.parse_args()

#reading in training data
fn_train='data/aggmean_cleaned_smalltrain.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

def runModel(i,j,featureCombo):
    mae = np.array([])   
    coef = None
    for ktrain,ktest in kf:
	predictor = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,featureCombo)]
	predictor2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,featureCombo)]
	out = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'Expected')]
	out2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'Expected')]
	predictor = predictor[out.Expected < 10]
	out = out[out.Expected < 10]
        model = linear_model.ElasticNet(alpha = i,l1_ratio=j,max_iter = 10000)
	#model = linear_model.SGDRegressor(alpha=i,loss='squared_loss',l1_ratio=j,penalty='elasticnet')
        model.fit(predictor,out.values)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('aggbymean,average 50-fold MAE for alpha %s l1_ratio %s feature %s: %s ' % (i,j,featureCombo,mae.mean()))
    logging.warning('coef: %s' % coef)

trainCleaned['KdpXradardist_km'] = trainCleaned['Kdp']*trainCleaned['radardist_km']
trainCleaned['RefXradardist_km'] = trainCleaned['Ref']*trainCleaned['radardist_km']
trainCleaned['RhoHVXradardist_km'] = trainCleaned['RhoHV']*trainCleaned['radardist_km']
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
for featureCombo in [args.f]:	
    #logging.warning('try feature %s' % featureCombo)
    allMAE = []
    jobs = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    	for i in [1e-10,1e-8,1e-6,1e-4,1e-3]:
		for j in [0.01,0.1,0.2,0.5,0.8]:
	    		runModel(i,j,featureCombo)
