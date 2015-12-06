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
#fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/2nd_submission.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData = pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV
cols = trainCleaned.columns
for i in cols:
    for j in cols:
        trainCleaned[str(i+'X'+j)] = (trainCleaned[i].values)*(trainCleaned[j].values)
        testData[str(i+'X'+j)] = (testData[i].values)*(testData[j].values)
#featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
featureCombo = trainCleaned.columns
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
model = linear_model.SGDClassifier(loss='squared_hinge',class_weight='auto',alpha=0.01,penalty='l1')
model.fit(trainCleaned[featureCombo],trainCleaned['class_exp'])
trainCleaned['predict_class_exp'] = model.predict(trainCleaned[featureCombo])
testData['predict_class_exp'] = model.predict(testData[featureCombo])
print('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
print('error rate:'+str(float(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp))/trainCleaned.shape[0]))
print('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))

#losshinge mean error rate 50-fold CV: 29.46%
#losslog mean error rate 50-fold CV: 22.92%
#lossmodified_huber mean error rate 50-fold CV: 23.86%
#losssquared_hinge mean error rate 50-fold CV: 18.88%
#lossperceptron mean error rate 50-fold CV: 26.14%

#model in 2 layers, 0~100, 100+
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
interaction = []
alpha = 1e-6
for i in featureCombo:
    newFeature = i+'X'+'predict_class_exp'
    trainCleaned[newFeature] = trainCleaned[i]*trainCleaned['predict_class_exp']
    testData[newFeature] = testData[i]*testData['predict_class_exp']
    interaction.append(newFeature)
featureCombo.extend(interaction)
featureCombo.append('predict_class_exp')

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
	predictor = predictor[out.Expected < 10]
	out = out[out.Expected < 10]
        model = linear_model.Lasso(alpha = i,max_iter=10000)
	#model = linear_model.SGDRegressor(loss='squared_loss',penalty='l1',alpha=i)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('coef: %s' % coef)
    logging.warning('aggByMean, average 50-fold MAE for alpha %s feature %s: %s ' % (i,featureCombo,mae.mean()))

logging.warning('start regression modeling')
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=False)
#for i in [1e-10,1e-8,1e-6,1e-4,1e-3]:
runModel(alpha,featureCombo)

#model = linear_model.Lasso(alpha)
#model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
#z = xModel.predict(y[featureCombo])
#z[z<0] = 0 #remove negative prediction
#ae = abs(allExp-allEst)
#ExpectedCut = pd.cut(allExp,[0,10,100,1e6])
#aeWithCut = pd.DataFrame({'ae':ae,'ExpectedCut':ExpectedCut})
#aeSum = aeWithCut.groupby(aeWithCut.ExpectedCut).agg(np.sum)
#logging.warning('ae by group %s' % aeSum)
