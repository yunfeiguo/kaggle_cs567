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
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
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
#trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
#testData.fillna(testData.mean().to_dict(),inplace=True)
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
def runModel(alpha,featureCombo):
    mae = np.array([])   
    logging.warning('try alpha = %s' % alpha)
#model in 4 layers, 0~1, 1~10, 10~100, >100
    for ktrain,ktest in kf:
        x = trainCleaned.iloc[ktrain,]
        y = trainCleaned.iloc[ktest,]    
        allEst = np.array([])
        allExp = np.array([])
        for i in [[0,1],[1,10],[10,100],[100,100000]]:
            subx = x[(x['Expected'] >= i[0]) & (x['Expected'] < i[1])]
            suby = y[(y['Expected'] >= i[0]) & (y['Expected'] < i[1])]
            subx.fillna(subx.mean().to_dict(),inplace=True)
            suby.fillna(suby.mean().to_dict(),inplace=True)
            subx = subx.groupby(subx.index).agg(np.mean)
            suby = suby.groupby(suby.index).agg(np.mean)
            #logging.warning('start modeling for interval %s' % i)
	    subx['KdpXradardist_km'] = subx['Kdp']*subx['radardist_km']
	    suby['KdpXradardist_km'] = suby['Kdp']*suby['radardist_km']
            xModel = linear_model.Lasso(alpha)
            xModel.fit(subx[featureCombo],subx['Expected'])
            #logging.warning('features: '+str(featureCombo))
            #logging.warning('coef: '+str(xModel.coef_))
            z = xModel.predict(suby[featureCombo])
            z[z<0] = 0 #remove negative prediction
            allEst = hstack([allEst,z])
            allExp = hstack([allExp,suby['Expected']])
        #mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
        mae = np.append(mae,getMAE(allEst,allExp))
    logging.warning('average 50-fold MAE for alpha %s feature %s' % (i,featureCombo))
    logging.warning(mae.mean())

#trainCleaned = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
#testData = testData.groupby(testData.index).agg(np.mean)
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=True)
#for featureCombo in [['RhoHV'],['RhoHV','Ref'],['RhoHV','Zdr'],['Ref','Kdp'],['Zdr','radardist_km'],['Kdp','radardist_km'],['KdpXradardist_km','Kdp','radardist_km'],['Kdp','Zdr','radardist_km']]:	
for featureCombo in [args.f]:	
    logging.warning('try feature %s' % featureCombo)
    allMAE = []
    jobs = []
    for i in frange(1,20,2):
	p = multiprocessing.Process(target=runModel, args=(i,featureCombo))
	jobs.append(p)
	p.start()
	if i % 3 == 0:
	    for j in jobs:
		j.join()
