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
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_1e7.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test_for_1e7.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
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
#trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
#testData.fillna(testData.mean().to_dict(),inplace=True)
#make sure there is no NaN any more
sum(np.isfinite(trainCleaned)==False) #return all zeros
sum(np.isfinite(testData)==False)
logging.warning('preprocessing done')

def meterological_est(ref, zdr, kdp, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    zh_aa = 0.027366
    zh_bb = 0.69444
    zzdr_aa = 0.00746
    zzdr_bb = 0.945
    zzdr_cc = -4.76
    kdp_aa = 40.6
    kdp_bb = 0.866
    kdpzdr_aa = 136
    kdpzdr_bb = 0.968
    kdpzdr_cc = -2.86
    mpsum = 0
    katsum = 0
    brsum = 0
    sacsum = 0
    ryzsum = 0
    for dbz, curzdr, curkdp, hours in zip(ref, zdr, kdp, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
	    dbz = pow(10,dbz/10)
	if np.isfinite(curzdr):
	    curzdr = pow(10,curzdr/10)
        if np.isfinite(dbz):
            mmperhr = pow(dbz/200, 0.625) #marshall palmer
            mpsum = mpsum + mmperhr * hours
            mmperhr = zh_aa * pow(dbz, zh_bb) #katsumata
            katsum = katsum + mmperhr * hours
        if np.isfinite(dbz) and np.isfinite(curzdr):
            mmperhr = zzdr_aa * pow(dbz, zzdr_bb) * pow(curzdr,zzdr_cc) #brandes
            brsum = brsum + mmperhr * hours
        if np.isfinite(curkdp):
            mmperhr = np.sign(curkdp) * kdp_aa * pow(abs(curkdp),kdp_bb) #sachidanandazrnic
            sacsum = sacsum + mmperhr * hours
        if np.isfinite(curkdp) and np.isfinite(curzdr):
	    mmperhr = np.sign(curkdp) * kdpzdr_aa * pow(abs(curkdp),kdpzdr_bb) * pow(curzdr,kdpzdr_cc) #Ryzhkov and Zrnic
            ryzsum = ryzsum + mmperhr * hours
    return [mpsum,katsum,brsum,sacsum,ryzsum]

# each unique Id is an hour of data at some gauge
def benchmarkFunc(hour):
    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    hour = hour.sort('minutes_past', ascending=True)
    est = pd.Series(meterological_est(hour['Ref'], hour['Zdr'], hour['Kdp'], hour['minutes_past']))
    return est
#MP must be calculated using unaggregated data
def transform (df,hasExpected):
    MP = df.groupby(df.index).apply(benchmarkFunc)
    MP.columns = ['MarshallPalmer','Katsumata','Brandes','Sachidanazrnic','RyzhkovZrnic']
    Expected = None
    if hasExpected:
        Expected = df.groupby(df.index).agg(np.mean)
        Expected = Expected['Expected']
    #trainCleaned = trainCleaned.groupby(trainCleaned.index).agg(np.sum)
    df = df.groupby(df.index).agg(np.mean)
    if hasExpected:
        df['Expected'] = Expected
    df = df.join(MP)
    return(df)

#model in 4 layers, 0~1, 1~10, 10~100, >100
allEst = np.array([])
allExp = np.array([])
for i in [[0,10],[10,100],[100,100000]]:
    x = trainCleaned[(trainCleaned['Expected'] >= i[0]) & (trainCleaned['Expected'] < i[1])]
    y = testData[(testData['Expected'] >= i[0]) & (testData['Expected'] < i[1])]
    x.fillna(x.mean().to_dict(),inplace=True)
    y.fillna(y.mean().to_dict(),inplace=True)
    x = transform(x,True)
    y = transform(y,True)
    x = x.groupby(x.index).agg(np.mean)
    y = y.groupby(y.index).agg(np.mean)
    featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
    alpha = 1e-6
    logging.warning('start modeling for interval %s' % i)
    xModel = linear_model.Lasso(alpha)
    xModel.fit(x[featureCombo],x['Expected'])
    logging.warning('features: '+str(featureCombo))
    logging.warning('coef: '+str(xModel.coef_))
    z = xModel.predict(y[featureCombo])
    z[z<0] = 0 #remove negative prediction
    allEst = hstack([allEst,z])
    allExp = hstack([allExp,y['Expected']])
#plt.yscale('log')
#plt.xscale('log')
#plt.scatter(allExp,allEst)
#plt.xlabel('Expcted')
#plt.ylabel('stratifyest')
#plt.savefig('largetrain_stratify_expected_scatter.png')
ae = abs(allExp-allEst)
ExpectedCut = pd.cut(allExp,[0,10,100,1e6])
aeWithCut = pd.DataFrame({'ae':ae,'ExpectedCut':ExpectedCut})
aeSum = aeWithCut.groupby(aeWithCut.ExpectedCut).agg(np.sum)
logging.warning('ae by group %s' % aeSum)
