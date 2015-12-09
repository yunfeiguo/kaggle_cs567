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

def runModel(i,featureCombo):
    mae = np.array([])   
    logging.warning('try alpha = %s' % i)
    coef = None
    for ktrain,ktest in kf:
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

trainCleaned['KdpXradardist_km'] = trainCleaned['Kdp']*trainCleaned['radardist_km']
trainCleaned['RefXradardist_km'] = trainCleaned['Ref']*trainCleaned['radardist_km']
trainCleaned['RhoHVXradardist_km'] = trainCleaned['RhoHV']*trainCleaned['radardist_km']
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
for featureCombo in [args.f]:	
    logging.warning('try feature %s' % featureCombo)
    allMAE = []
    jobs = []
    for i in [0.001,0.1,0.5,1,2,4,8,12,20]:
	p = multiprocessing.Process(target=runModel, args=(i,featureCombo))
	jobs.append(p)
	p.start()
	if i % 2 == 0:
	    for j in jobs:
		j.join()
