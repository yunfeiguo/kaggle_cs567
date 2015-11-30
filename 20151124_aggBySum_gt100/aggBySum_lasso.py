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
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
#fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/2nd_submission.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
#trainCleaned=dd.read_csv(fn_train,sep=',')
#trainCleaned.set_index('Id')
n=len(trainCleaned.columns)
logging.warning('reading done')

#drop records whose Ref columns are all NaNs
trainCleaned = trainCleaned.groupby(trainCleaned.index).filter(lambda x: sum(np.isfinite(x['Ref'])) > 1)
#fill remaining NaNs with column mean
#alternatively, try group mean
trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
#make sure there is no NaN any more
sum(np.isfinite(trainCleaned)==False) #return all zeros
logging.warning('preprocessing done')


def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum


# each unique Id is an hour of data at some gauge
def benchmarkFunc(hour):
    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est

##also, write a function for generating benchmark MAE
#est = testCleaned.groupby(testCleaned.index).apply(getBenchMark)
#print('benchmark MAE')
#print(getMAE(est,testNoAgg['Expected']))
#split the data by distance categories (perhaps use 10km bins)
#then do regression within each bin
#see if interaction effects exist
def runModel(i,featureCombo):
    mae = np.array([])   
    logging.warning('try alpha = %s' % i)
    coef = None
    for ktrain,ktest in kf:
        #x = trainCleaned.iloc[ktrain,]
        #y = trainCleaned.iloc[ktest,]    
	#x = x[x.Expected < 10]
        #model = linear_model.Lasso(alpha = i)
        #model.fit(x[featureCombo],x['Expected'])
        #mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
	predictor = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,featureCombo)]
	predictor2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,featureCombo)]
	out = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'Expected')]
	out2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'Expected')]
	predictor = predictor[out.Expected < 100]
	out = out[out.Expected < 100]
        model = linear_model.Lasso(alpha = i)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('coef: %s' % coef)
    logging.warning('average 50-fold MAE for alpha %s feature %s: %s ' % (i,featureCombo,mae.mean()))

trainCleaned['KdpXradardist_km'] = trainCleaned['Kdp']*trainCleaned['radardist_km']
trainCleaned['RefXradardist_km'] = trainCleaned['Ref']*trainCleaned['radardist_km']
trainCleaned['RhoHVXradardist_km'] = trainCleaned['RhoHV']*trainCleaned['radardist_km']
#MP must be calculated using unaggregated data
MP = trainCleaned.groupby(trainCleaned.index).apply(benchmarkFunc)
Expected = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
Expected = Expected['Expected']
trainCleaned = trainCleaned.groupby(trainCleaned.index).agg(np.sum)
#trainCleaned = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
trainCleaned['MarshallPalmer'] = MP
trainCleaned['Expected'] = Expected
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=False)
#for featureCombo in [['RhoHV'],['RhoHV','Ref'],['RhoHV','Zdr'],['Ref','Kdp'],['Zdr','radardist_km'],['Kdp','radardist_km'],['KdpXradardist_km','Kdp','radardist_km'],['Kdp','Zdr','radardist_km']]:	
for featureCombo in [args.f]:	
    logging.warning('try feature %s' % featureCombo)
    allMAE = []
    jobs = []
    for i in [0.001,0.1,0.5,1,2,4,8,12,20]:
	p = multiprocessing.Process(target=runModel, args=(i,featureCombo))
	jobs.append(p)
	p.start()
	if i % 3 == 0:
	    for j in jobs:
		j.join()
    	#mae = np.array([])   
   	#logging.warning('try alpha = %s' % i)
    	#for ktrain,ktest in kf:
    	#    ktrainData = trainCleaned.iloc[ktrain,]
    	#    ktestData = trainCleaned.iloc[ktest,]    
    	#    x = ktrainData.groupby(ktrainData.index).agg(np.mean)
    	#    y = ktestData.groupby(ktestData.index).agg(np.mean)
    	#    model = linear_model.Lasso(alpha = i)
    	#    model.fit(x[featureCombo],x['Expected'])
    	#    mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
    	#allMAE.append(mae.mean())
    #logging.warning('average 10-fold MAE for alpha=10~20')
    #logging.warning(allMAE)

#x = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
#y = testData.groupby(testData.index).agg(np.mean)
#logging.warning('start modeling')
#featureCombo = ['radardist_km','Kdp']
#alpha = 0.9
#xModel = linear_model.Lasso(alpha)
#xModel.fit(x[featureCombo],x['Expected'])
#logging.warning('features: '+str(featureCombo))
#logging.warning('coef: '+str(xModel.coef_))
#z = pd.DataFrame(pd.DataFrame.dot(y[featureCombo],xModel.coef_))
#z.columns = ['Expected']
#z.to_csv(fn_out)
#logging.warning('Prediction done')
#logging.warning('Gzipping...')
#os.system('gzip -f '+fn_out)
#logging.warning('Output written to '+fn_out+'.gz')
