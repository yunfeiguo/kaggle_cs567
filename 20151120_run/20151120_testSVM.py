#!/usr/bin/env python
import os
os.chdir("/home/yunfeiguo/projects/kaggle_cs567/src")
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import dask.dataframe as dd
from yg_utils import *
from sklearn import linear_model
from sklearn import svm
import logging
import argparse
import sys
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

parser = argparse.ArgumentParser()
parser.add_argument('-f',nargs='+',required=True)
args = parser.parse_args()


#reading in training data
fn_train='../data/train.csv'
fn_test='../data/test.csv'
#fn_train='../data/train_100000.csv'
#fn_test='../data/smalltest.v2.csv'
trainData=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
#trainData=dd.read_csv(fn_train,sep=',')
#trainData.set_index('Id')
#testData=dd.read_csv(fn_test,sep=',')
#testData.set_index('Id')
n=len(trainData.columns)
logging.warning('reading done')

#drop records whose Ref columns are all NaNs
trainCleaned = trainData.groupby(trainData.index).filter(lambda x: sum(np.isfinite(x['Ref'])) > 1)
#fill remaining NaNs with column mean
#alternatively, try group mean
trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
testData.fillna(testData.mean().to_dict(),inplace=True)
#make sure there is no NaN any more
sum(np.isfinite(trainCleaned)==False) #return all zeros
sum(np.isfinite(testData)==False)
logging.warning('preprocessing done')

##also, write a function for generating benchmark MAE
#est = testCleaned.groupby(testCleaned.index).apply(getBenchMark)
#print('benchmark MAE')
#print(getMAE(est,testNoAgg['Expected']))
#split the data by distance categories (perhaps use 10km bins)
#then do regression within each bin
#see if interaction effects exist
def runModel(e,c,k,featureCombo):
    mae = np.array([])   
    for ktrain,ktest in kf:
        x = trainCleaned.iloc[ktrain,]
        y = trainCleaned.iloc[ktest,]    
	model = svm.SVR(kernel=k,C=c,epsilon=e,max_iter=20000)
        model.fit(x[featureCombo],x['Expected'])
        mae = np.append(mae,(getMAE(model.predict(y[featureCombo]),y['Expected'])))
    logging.warning('average MAE for kernel %s c %s epsilon %s feature %s: %s' % (k,c,e,featureCombo,mae.mean()))

trainCleaned['KdpXradardist_km'] = trainCleaned['Kdp']*trainCleaned['radardist_km']
trainCleaned = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
testData = testData.groupby(testData.index).agg(np.mean)
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
#for featureCombo in [['RhoHV'],['RhoHV','Ref'],['RhoHV','Zdr'],['Ref','Kdp'],['Zdr','radardist_km'],['Kdp','radardist_km'],['KdpXradardist_km','Kdp','radardist_km'],['Kdp','Zdr','radardist_km']]:	
count = 0
for featureCombo in [args.f]:	
    logging.warning('try feature %s' % featureCombo)
    allMAE = []
    jobs = []
    for e in [0.001,0.1,0.1,1]:
	for c in [0.001,0.01,0.1,1]:
	    for k in ['rbf','linear','poly','sigmoid']:
		p = multiprocessing.Process(target=runModel, args=(e,c,k,featureCombo))
		jobs.append(p)
		p.start()
		count += 1
		if count % 6 == 0:
	    	    for j in jobs:
			j.join()
    	#mae = np.array([])   
   	#logging.warning('try alpha = %s' % i)
    	#for ktrain,ktest in kf:
    	#    ktrainData = trainCleaned.iloc[ktrain,]
    	#    ktestData = trainCleaned.iloc[ktest,]    
    	#    x = ktrainData.groupby(ktrainData.index).agg(np.mean)
    	#    y = ktestData.groupby(ktestData.index).agg(np.mean)
    	#    model = linear_model.Lasso(alpha = i)
    	#    model.fit(x[featureCombo],x['Expected'])
    	#    mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
    	#allMAE.append(mae.mean())
    #logging.warning('average 10-fold MAE for alpha=10~20')
    #logging.warning(allMAE)
