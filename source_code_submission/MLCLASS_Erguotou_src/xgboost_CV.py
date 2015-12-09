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
