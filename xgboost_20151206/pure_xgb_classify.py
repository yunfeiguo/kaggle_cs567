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
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_manyfeature_xgb.txt'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_manyfeature_600k_xgb.txt'
fn_eval='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test_manyfeature_600k_xgb.txt'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test_manyfeature_xgb.txt'
fn_alltrain='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_manyfeature_xgb.txt'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_70k_xgb.v2.txt'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_600k_xgb.v2.txt'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test_600k_xgb.v2.txt'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
#fn_train = '/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain_xgb.v2.txt'
#testData = pd.read_csv(fn_test,sep=',',index_col='Id')
trainCleaned = xgb.DMatrix(fn_train)
evalData = xgb.DMatrix(fn_eval)
testData = xgb.DMatrix(fn_test)
alltrain = xgb.DMatrix(fn_alltrain)
logging.warning('reading done')

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)
#param = {'max_depth':200, 'eta':0.3, 'silent':0, 'objective':'binary:logistic'}
param = {'max_depth':20, 'eta':0.3, 'silent':0, 'objective':'reg:linear'} #2^17=130k
num_round = 10
#logging.warning(xgb.cv(param, alltrain, num_round, nfold=50, metrics={'rmse'}, seed = 0, fpreproc = fpreproc))
logging.warning(xgb.cv(param, alltrain, num_round, nfold=50, metrics={'rmse'}, seed = 0))
#num_round = 100
#evallist  = [(evalData,'eval'), (trainCleaned,'train')]
#bst =xgb.train(param,trainCleaned,evals=evallist, early_stopping_rounds=20)
#preds = bst.predict(testData)
#labels = testData.get_label()
#logging.warning(getMAE(labels,preds))
#print('test')
#for i in preds:
#    if i < 0:
#	print(0)
#    else:
#        print(i)
#print('alltrain')
#for i in bst.predict(alltrain):
#    if i < 0:
#	print(0)
#    else:
#        print(i)
#if len(preds) > 0:
#    logging.warning ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
#class1error = 0
#class1count = 0
#for i in range(len(preds)):
#    if labels[i] == 1:
#	if preds[i] <= 0.5:
#	    class1error += 1
#	class1count += 1
#if class1count > 0:	
#    logging.warning ('class1error=%f' % (class1error/float(class1count)))
