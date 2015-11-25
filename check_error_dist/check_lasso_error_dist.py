#!/usr/bin/env python
import os
import sys
sys.path.append("/home/yunfeiguo/projects/kaggle_cs567/src")
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pylab import *
import matplotlib.pyplot as plt
import dask.dataframe as dd
from yg_utils import *
from sklearn import linear_model
import logging
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_1e7.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test_for_1e7.csv'
#fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/2nd_submission.csv'
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



x = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
y = testData.groupby(testData.index).agg(np.mean)
logging.warning('start modeling')
featureCombo = ['RhoHV','Zdr']
alpha = 1
xModel = linear_model.Lasso(alpha)
xModel.fit(x[featureCombo],x['Expected'])
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(xModel.coef_))
z = pd.DataFrame(pd.DataFrame.dot(y[featureCombo],xModel.coef_))
#z = xModel.predict(y[featureCombo])
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
plt.yscale('log')
plt.xscale('log')
plt.scatter(y['Expected'],z['Expected'])
plt.xlabel('Expcted')
plt.ylabel('lassoest')
plt.savefig('largetrain_lassoest_expected_scatter.png')
