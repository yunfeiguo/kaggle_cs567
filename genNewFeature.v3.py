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
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.v2.csv'
#fn_out_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain_manyfeature.csv'
#fn_out_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest_manyfeature.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
fn_out_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_manyfeature.csv'
fn_out_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test_manyfeature.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

cols = trainCleaned.columns[trainCleaned.columns != 'Expected']
for i in range(len(cols)):
    for j in range(i,len(cols)):
	newFeature = str(cols[i]+'X'+cols[j])
        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
        testData[newFeature] = testData[cols[i]]*testData[cols[j]]
#add square terms
for i in range(len(cols)):
    newFeature = str(cols[i]+'2')
    trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[i]]
    testData[newFeature] = testData[cols[i]]*testData[cols[i]]
#add log terms
for i in range(len(cols)):
    newFeature = str(cols[i]+'log')
    trainCleaned[newFeature] = np.log(trainCleaned[cols[i]])
    testData[newFeature] = np.log(testData[cols[i]])
#add inverse
for i in range(len(cols)):
    newFeature = str(cols[i]+'inverse')
    trainCleaned[newFeature] = 1./trainCleaned[cols[i]]
    testData[newFeature] = 1./testData[cols[i]]

trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
testData.fillna(testData.mean().to_dict(),inplace=True)

trainCleaned.to_csv(fn_out_train)
testData.to_csv(fn_out_test)
logging.warning('Output done')
