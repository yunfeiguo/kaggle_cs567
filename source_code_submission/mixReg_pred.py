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
import argparse
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('started')

#reading in training data
fn_train='data/aggmean_cleaned_smalltrain.csv'
fn_test='data/aggmean_cleaned_smalltest.csv'
fn_out_test = 'data/mixReg_pred.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData = pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

#####################KNN############################
from sklearn import neighbors as nb
cols = trainCleaned.columns[trainCleaned.columns!='Expected']
classification_featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
classification_model = nb.KNeighborsClassifier(20,weights='distance',n_jobs=31,p=1)
classification_model.fit(trainCleaned[classification_featureCombo],trainCleaned['class_exp'])
trainCleaned['predict_class_exp'] = classification_model.predict(trainCleaned[classification_featureCombo])
testData['predict_class_exp'] = classification_model.predict(testData[classification_featureCombo])
print('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
print('error rate:'+str(float(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp))/trainCleaned.shape[0]))
print('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))
#model in 2 layers, 0~100, 100+
logging.warning('start regression modeling')

trainCleaned = trainCleaned.drop('predict_class_exp',1)
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
featureCombo_for_test = featureCombo[:]
interaction = []
interaction_for_test = []
for i in featureCombo:
    newFeature = i+'X'+'class_exp'
    newFeature_for_test = i+'X'+'predict_class_exp'
    trainCleaned[newFeature] = trainCleaned[i]*(trainCleaned.class_exp.astype('int'))
    testData[newFeature_for_test] = testData[i]*testData['predict_class_exp']
    interaction.append(newFeature)
    interaction_for_test.append(newFeature_for_test)
featureCombo.extend(interaction)
featureCombo.append('class_exp')
featureCombo_for_test.extend(interaction_for_test)
featureCombo_for_test.append('predict_class_exp')
logging.warning('start regression modeling')
trainCleaned['class_exp'] = trainCleaned.class_exp.astype('int')
alpha = 1e-3
model = linear_model.Lasso(alpha = alpha,max_iter=100000)
model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
logging.warning('features: '+str(featureCombo_for_test))
logging.warning('coef: '+str(model.coef_))
z = pd.DataFrame(model.predict(testData[featureCombo_for_test]),index=testData.index)
logging.warning('Prediction done')
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out_test)
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out_test)
logging.warning('Output written to '+fn_out_test+'.gz')
