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
from sklearn import svm
import logging
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_1e7.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test_for_1e7.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/20151206_aggmean_lasso_manyfeature.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.v2.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData = pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV
cols = trainCleaned.columns[trainCleaned.columns!='Expected']
for i in range(len(cols)):
    for j in range(i,len(cols)):
        newFeature = str(cols[i]+'X'+cols[j])
        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
        testData[newFeature] = testData[cols[i]]*testData[cols[j]]
featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']

#model in 2 layers, 0~100, 100+
logging.warning('start regression modeling')
alpha = 1e-4
model = linear_model.Lasso(alpha = alpha,max_iter=10000)
model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
z = pd.DataFrame(model.predict(testData[featureCombo]),index=testData.index)
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(model.coef_))
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out)
logging.warning('Prediction done')
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
