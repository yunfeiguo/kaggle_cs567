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
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

##model in 4 layers, 0~1, 1~10, 10~100, >100
#allEst = np.array([])
#allExp = np.array([])
#for i in [[0,10],[10,100],[100,100000]]:
#    x = trainCleaned[(trainCleaned['Expected'] >= i[0]) & (trainCleaned['Expected'] < i[1])]
#    y = testData[(testData['Expected'] >= i[0]) & (testData['Expected'] < i[1])]
#    x.fillna(x.mean().to_dict(),inplace=True)
#    y.fillna(y.mean().to_dict(),inplace=True)
#    x = transform(x,True)
#    y = transform(y,True)
#    x = x.groupby(x.index).agg(np.mean)
#    y = y.groupby(y.index).agg(np.mean)
#    featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
#    alpha = 1e-6
#    logging.warning('start modeling for interval %s' % i)
#    xModel = linear_model.Lasso(alpha)
#    xModel.fit(x[featureCombo],x['Expected'])
#    logging.warning('features: '+str(featureCombo))
#    logging.warning('coef: '+str(xModel.coef_))
#    z = xModel.predict(y[featureCombo])
#    z[z<0] = 0 #remove negative prediction
#    allEst = hstack([allEst,z])
#    allExp = hstack([allExp,y['Expected']])
##plt.yscale('log')
##plt.xscale('log')
##plt.scatter(allExp,allEst)
##plt.xlabel('Expcted')
##plt.ylabel('stratifyest')
##plt.savefig('largetrain_stratify_expected_scatter.png')
#ae = abs(allExp-allEst)
#ExpectedCut = pd.cut(allExp,[0,10,100,1e6])
#aeWithCut = pd.DataFrame({'ae':ae,'ExpectedCut':ExpectedCut})
#aeSum = aeWithCut.groupby(aeWithCut.ExpectedCut).agg(np.sum)
#logging.warning('ae by group %s' % aeSum)

# classification by logistic and learn parameter by CV

#X = trainCleaned.fillna(x.mean().to_dict(),inplace=True)
#X = transform(X,True)
#X = X.groupby(X.index).agg(np.mean)
X = trainCleaned
class_exp=pd.cut(X['Expected'],[0,100,1e6],labels=[0,1])
X['class_exp']=class_exp
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']

kf = KFold(X.shape[0],n_folds=50,shuffle=False)
error_percent = np.array([])   
for ktrain,ktest in kf:
    ktrainData = X.iloc[ktrain,]
    ktestData = X.iloc[ktest,]    
    #x = ktrainData.groupby(ktrainData.index).agg(np.mean)
    #y = ktestData.groupby(ktestData.index).agg(np.mean)
    #featureCombo = ['Kdp','radardist_km']
    model = linear_model.SGDClassifier()
    model.fit(ktrainData[featureCombo],ktrainData['class_exp'])
    #cur_error_percent=(sum(abs(model.predict(ktestData[featureCombo])-np.array(ktestData['class_exp'])))*100/ktestData.shape[0],3)
    cur_error_percent=sum(abs(model.predict(ktestData[featureCombo])-np.array(ktestData['class_exp'])))*100/ktestData.shape[0]
    error_percent=np.append(error_percent,cur_error_percent)
print('mean error rate 50-fold CV: '+str(error_percent.mean())+'%') # note: this number is percentage


trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
cols = trainCleaned.columns
for i in cols:
    for j in cols:
        trainCleaned[str(i+'X'+j)] = (trainCleaned[i].values)*(trainCleaned[j].values)
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
featureCombo = trainCleaned.columns
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
model = linear_model.SGDClassifier(class_weight='auto',alpha=0.1,penalty='l1')
model.fit(trainCleaned[featureCombo],trainCleaned['class_exp'])
trainCleaned['predict_class_exp'] = model.predict(trainCleaned[featureCombo])
print('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
print('total wrong predictions:'+str(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp)))
print('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))

kf = KFold(X.shape[0],n_folds=50,shuffle=False)
for loss in ['hinge','log','modified_huber','squared_hinge','perceptron']:
    error_percent = np.array([])   
    for ktrain,ktest in kf:
        ktrainData = trainCleaned.iloc[ktrain,]
        ktestData = trainCleaned.iloc[ktest,]    
        model = linear_model.SGDClassifier(loss=loss,class_weight='auto',alpha=0.1,penalty='l1')
        model.fit(ktrainData[featureCombo],ktrainData['class_exp'])
        cur_error_percent=sum(abs(model.predict(ktestData[featureCombo])-np.array(ktestData['class_exp'])))*100/ktestData.shape[0]
        error_percent=np.append(error_percent,cur_error_percent)
    print('loss'+loss+' mean error rate 50-fold CV: '+str(error_percent.mean())+'%') # note: this number is percentage
#losshinge mean error rate 50-fold CV: 29.46%
#losslog mean error rate 50-fold CV: 22.92%
#lossmodified_huber mean error rate 50-fold CV: 23.86%
#losssquared_hinge mean error rate 50-fold CV: 18.88%
#lossperceptron mean error rate 50-fold CV: 26.14%

featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
featureCombo = trainCleaned.columns
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
model = svm.SVC(class_weight='auto')
model.fit(trainCleaned[featureCombo],trainCleaned['class_exp'])
trainCleaned['predict_class_exp'] = model.predict(trainCleaned[featureCombo])
print('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
print('total wrong predictions:'+str(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp)))
print('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))

kf = KFold(X.shape[0],n_folds=50,shuffle=False)
for kernel in ['linear','poly','rbf','sigmoid']:
    for c in [0.1,1,10]:
        error_percent = np.array([])   
        for ktrain,ktest in kf:
            ktrainData = trainCleaned.iloc[ktrain,]
            ktestData = trainCleaned.iloc[ktest,]    
            model = svm.SVC(C=c,kernel=kernel,class_weight='auto')
            model.fit(ktrainData[featureCombo],ktrainData['class_exp'])
            cur_error_percent=sum(abs(model.predict(ktestData[featureCombo])-np.array(ktestData['class_exp'])))*100/ktestData.shape[0]
            error_percent=np.append(error_percent,cur_error_percent)
        print('kernel '+loss+'C '+str(c)+' mean error rate 50-fold CV: '+str(error_percent.mean())+'%') # note: this number is percentage
