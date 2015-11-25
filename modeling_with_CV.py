from sklearn.cross_validation import KFold
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import dask.dataframe as dd
from yg_utils import *
from sklearn import linear_model 

#reading in training data
os.chdir("E:/kaggle_cs567/try_LR")
fn_train='../data/train_100000.csv'
fn_test='../data/smalltest.csv'
trainData=pd.read_csv(fn_train,sep=',',index_col='Id')
testData=pd.read_csv(fn_test,sep=',',index_col='Id')
n=len(trainData.columns)

#drop records whose Ref columns are all NaNs
trainCleaned = trainData.groupby(trainData.index).filter(lambda x: sum(np.isfinite(x['Ref'])) > 1)
#fill remaining NaNs with column mean
#alternatively, try group mean
trainCleaned.fillna(trainCleaned.mean().to_dict(),inplace=True)
testCleaned.fillna(testCleaned.mean().to_dict(),inplace=True)

kf = KFold(trainCleaned.shape[0],n_folds=10,shuffle=True)
allMAE = []
for i in frange(0,10,0.5):
    mae = np.array([])   
    for ktrain,ktest in kf:
        ktrainData = trainCleaned.iloc[ktrain,]
        ktestData = trainCleaned.iloc[ktest,]    
        x = ktrainData.groupby(ktrainData.index).agg(np.mean)
        y = ktestData.groupby(ktestData.index).agg(np.mean)
        featureCombo = ['Kdp','radardist_km']
        model = linear_model.Lasso(alpha = i)
        model.fit(x[featureCombo],x['Expected'])
        mae = np.append(mae,(getMAE(np.dot(y[featureCombo],model.coef_),y['Expected'])))
    allMAE.append(mae.mean())
print(allMAE)

#x = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
#y = testCleaned.groupby(testCleaned.index).agg(np.mean)
#x['KdpXradardist_km'] = x['Kdp']*x['radardist_km']
#y['KdpXradardist_km'] = y['Kdp']*y['radardist_km']
##expected < 10
#x10 = x[x['Expected'] < 10]
#y10 = y[y['Expected'] < 10]
##expected >= 10, < 100
#x100 = x[(x['Expected'] >= 10) & (x['Expected'] < 100)]
#y100 = y[(y['Expected'] >= 10) & (y['Expected'] < 100)]
##expected >= 100
#x100plus = x[x['Expected'] > 100]
#y100plus = y[y['Expected'] > 100]
##linear elastic net regression
#for tryData in [[x10,y10],[x100,y100],[x100plus,y100plus]]:
#    for featureCombo in [['RhoHV'],['RhoHV','Ref'],['RhoHV','Zdr'],['Ref','Kdp'],['Zdr','radardist_km'],['Kdp','radardist_km'],['KdpXradardist_km','Kdp','radardist_km'],['Kdp','Zdr','radardist_km']]:
#        print('features:'+str(featureCombo))
#        for i in frange(0,1,0.1):
#            print('alpha:'+str(i))
#            #xElasticNet = linear_model.ElasticNet (alpha = i)
#            #xElasticNet = linear_model.Lasso(alpha = i)
#            xElasticNet = linear_model.SGDRegressor(alpha = i, l1_ratio = 0.15, loss = 'huber')
#            xElasticNet.fit(x[featureCombo],x['Expected'])
#            if len(featureCombo) == 1:
#                getMAE(y[featureCombo]*xElasticNet.coef_,y[['Expected']])
#            else:            
#                getMAE(np.dot(y[featureCombo],xElasticNet.coef_),y['Expected'])          
