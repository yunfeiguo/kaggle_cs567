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
#parser.add_argument('-f',nargs='+',required=True)
#parser.add_argument('-l',nargs='+',required=True)
args = parser.parse_args()

#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_boostRidgeLassoEN.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_boostRigdeLassoENRansacBayesMixregXgbRegRF.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test_boostRigdeLassoENRansacBayesMixregXgbRegRF.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain_boostRigdeLassoENRansacBayesMixregXgbRegRF.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest_boostRigdeLassoENRansacBayesMixregXgbRegRF.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
fn_out ='/home/yunfeiguo/projects/kaggle_cs567/results/20151207_ensemble_submission_lassoXGB_a1e-4.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

cutoff = 15
#featureCombo = [ 'boost_pred', 'linearRidge_pred', 'lassoPred', 'ElasticNetPred', 'ransac_pred', 'Bay_pred', 'mixregPred', 'RF_pred', 'xgbPred']
#trainCleaned['lassoPred2'] = 
featureCombo = [  'lassoPred','lassoPred2']
alpha = 0.1

logging.warning('try feature %s' % featureCombo)
trainCleaned = trainCleaned[trainCleaned.Expected < cutoff]
logging.warning('start modeling')
model = linear_model.Lasso(alpha,max_iter=10000)
model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(model.coef_))
z = pd.DataFrame(pd.DataFrame.dot(testData[featureCombo],model.coef_))
#z = xModel.predict(y[featureCombo])
z.columns = ['Expected']
z.Expected[z.Expected<0] = 0 #remove negative prediction
z.to_csv(fn_out)
logging.warning('Prediction done')
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
