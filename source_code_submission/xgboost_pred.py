#!/usr/bin/env python
import os
import sys
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
import xgboost as xgb
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

parser = argparse.ArgumentParser()
#parser.add_argument('-k',nargs='+',required=True)
#parser.add_argument('-p',nargs='+',required=True)
#parser.add_argument('-a',nargs='+',required=True)
#parser.add_argument('-w',nargs='+',required=True)
args = parser.parse_args()

#reading in training data
fn_train='data/aggmean_cleaned_smalltrain_manyfeature_xgb.txt'
trainCleaned = xgb.DMatrix(fn_train)
logging.warning('reading done')

param = {'max_depth':20, 'eta':0.3, 'silent':0, 'objective':'reg:linear'} #2^17=130k
num_round = 1
logging.warning(xgb.cv(param, trainCleaned, num_round, nfold=50, metrics={'rmse'}, seed = 0))



#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_manyfeature_xgb.txt'
fn_alltrain='data/aggmean_cleaned_smalltrain_manyfeature_xgb.txt'
fn_train='data/aggmean_cleaned_smalltrain_manyfeature_xgb_part1.txt'
fn_eval='data/aggmean_cleaned_smalltrain_manyfeature_xgb_part2.txt'
fn_test='data/aggmean_cleaned_smalltest_manyfeature_xgb.txt'
fn_test_csv='data/aggmean_cleaned_smalltest_manyfeature.csv'
fn_out = 'data/xgboost_pred.csv'
trainCleaned = xgb.DMatrix(fn_train)
evalData = xgb.DMatrix(fn_eval)
testData = xgb.DMatrix(fn_test)
alltrain = xgb.DMatrix(fn_alltrain)
logging.warning('reading done')

#param = {'max_depth':200, 'eta':0.3, 'silent':0, 'objective':'binary:logistic'}
param = {'max_depth':20, 'eta':0.3, 'silent':0, 'objective':'reg:linear'} #2^17=130k
num_round = 10
evallist  = [(evalData,'eval'), (trainCleaned,'train')]
bst =xgb.train(param,trainCleaned,evals=evallist, early_stopping_rounds=20)
preds = bst.predict(testData)
logging.warning('Prediction done')

#write output
f = open(fn_out,'w')
testf = open(fn_test_csv,'r')
header = testf.readline()
f.write('Id,Expected\n')
for i in preds:
    fields = testf.readline().split(',')
    id = fields[0]
    if i < 0:
	f.write(str(id)+',0')
    else:
        f.write(str(id)+','+str(i))
    f.write('\n')
f.close()
testf.close()
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
