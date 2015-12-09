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
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV

trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
#cols = trainCleaned.columns[trainCleaned.columns!='Expected']
#for i in range(len(cols)):
#    for j in range(i,len(cols)):
#        newFeature = str(cols[i]+'X'+cols[j])
#        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
#        #testData[newFeature] = testData[cols[i]]*testData[cols[j]]
#featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
classification_featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
#model = linear_model.SGDClassifier(class_weight='auto',alpha=0.1,penalty='l1')
#model.fit(trainCleaned[classification_featureCombo],trainCleaned['class_exp'])
#trainCleaned['predict_class_exp'] = model.predict(trainCleaned[classification_featureCombo])
#print('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
#print('total wrong predictions:'+str(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp)))
#print('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))

############SGDClassifier#####################
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=False)
for loss in ['hinge','log','modified_huber','squared_hinge','perceptron']:
    error_percent = np.array([])   
    class1_error_percent = np.array([])   
    for ktrain,ktest in kf:
        ktrainData = trainCleaned.iloc[ktrain,]
        ktestData = trainCleaned.iloc[ktest,]    
        model = linear_model.SGDClassifier(loss=loss,class_weight='balanced',alpha=0.1,penalty='l1')
        model.fit(ktrainData[classification_featureCombo],ktrainData['class_exp'])
	prediction = model.predict(ktestData[classification_featureCombo])
        cur_error_percent=sum(prediction != ktestData['class_exp'])*100.0/ktestData.shape[0]
        cur_class1error=sum(ktestData.class_exp[prediction != ktestData['class_exp']].values)
        error_percent=np.append(error_percent,cur_error_percent)
        class1_error_percent=np.append(class1_error_percent,cur_class1error)
    print('loss'+loss+' mean error rate 50-fold CV: '+str(error_percent.mean())+'%') # note: this number is percentage
    print('loss'+loss+' mean class1 error rate 50-fold CV: '+str(class1_error_percent.sum()*100.0/sum(trainCleaned.class_exp.values))+'%') # note: this number is percentage
#####################################################
#losshinge mean error rate 50-fold CV: 29.46%
#losslog mean error rate 50-fold CV: 22.92%
#lossmodified_huber mean error rate 50-fold CV: 23.86%
#losssquared_hinge mean error rate 50-fold CV: 18.88%
#lossperceptron mean error rate 50-fold CV: 26.14%

######################KNN############################
#from sklearn import neighbors as nb
#kf = KFold(trainCleaned.shape[0],n_folds=10,shuffle=False)
#K = 5
#for p in [1,2,3,5]:
#    error_percent = np.array([])   
#    class1_error_percent = np.array([])   
#    for ktrain,ktest in kf:
#        ktrainData = trainCleaned.iloc[ktrain,]
#        ktestData = trainCleaned.iloc[ktest,]    
#        #model = linear_model.SGDClassifier(loss=loss,class_weight='auto',alpha=0.1,penalty='l1')
#	model = nb.KNeighborsClassifier(K,weights='uniform',n_jobs=31,p=p)
#        model.fit(ktrainData[classification_featureCombo],ktrainData['class_exp'])
#	prediction = model.predict(ktestData[classification_featureCombo])
#        cur_error_percent=sum(prediction != ktestData['class_exp'])*100.0/ktestData.shape[0]
#        cur_class1error=sum(ktestData.class_exp[prediction != ktestData['class_exp']].values)
#        error_percent=np.append(error_percent,cur_error_percent)
#        class1_error_percent=np.append(class1_error_percent,cur_class1error)
#    print('p '+str(p)+' K'+str(K)+' mean error rate 50-fold CV: '+str(error_percent.mean())+'%') # note: this number is percentage
#    print('p '+str(p)+' K'+str(K)+' mean class1 error rate 50-fold CV: '+str(class1_error_percent.sum()*100.0/sum(trainCleaned.class_exp.values))+'%') # note: this number is percentage
