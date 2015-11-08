#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import dask.dataframe as dd
from yg_utils import *
from sklearn import linear_model
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

#reading in training data
os.chdir("/home/yunfeiguo/projects/kaggle_cs567/src")
fn_train='../data/train.csv'
fn_test='../data/test.csv'
#fn_train='../data/train_100000.csv'
#fn_test='../data/smalltest.v2.csv'
fn_out='../results/2nd_submission.csv'
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

x = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
y = testData.groupby(testData.index).agg(np.mean)
logging.warning('start modeling')
featureCombo = ['radardist_km','Kdp']
alpha = 0.9
xModel = linear_model.Lasso(alpha)
xModel.fit(x[featureCombo],x['Expected'])
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(xModel.coef_))
z = pd.DataFrame(pd.DataFrame.dot(y[featureCombo],xModel.coef_))
z.columns = ['Expected']
z.to_csv(fn_out)
logging.warning('Prediction done')
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
