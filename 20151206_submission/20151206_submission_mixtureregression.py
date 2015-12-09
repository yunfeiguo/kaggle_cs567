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
fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/20151206_mixture_knn_k5_p2_alpha1e-3.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
testData = pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV
logging.warning('start classification')
cols = trainCleaned.columns[trainCleaned.columns!='Expected']
for i in range(len(cols)):
    for j in range(i,len(cols)):
        newFeature = str(cols[i]+'X'+cols[j])
        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
        testData[newFeature] = testData[cols[i]]*testData[cols[j]]
classification_featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])

from sklearn import neighbors as nb
kf = KFold(trainCleaned.shape[0],n_folds=10,shuffle=False)
K = 5
p = 2
model = nb.KNeighborsClassifier(K,weights='uniform',p=p)
model.fit(trainCleaned[classification_featureCombo],trainCleaned['class_exp'])
trainCleaned['predict_class_exp'] = model.predict(trainCleaned[classification_featureCombo])
testData['predict_class_exp'] = model.predict(testData[classification_featureCombo])
cur_error_percent=sum(trainCleaned['predict_class_exp'] != trainCleaned['class_exp'])*100.0/trainCleaned.shape[0]
cur_class1error=sum(trainCleaned.class_exp[trainCleaned['predict_class_exp'] != trainCleaned['class_exp']].values)
logging.warning('p '+str(p)+' K'+str(K)+' mean error rate : '+str(cur_error_percent)+'%') # note: this number is percentage
logging.warning('p '+str(p)+' K'+str(K)+' mean class1 error rate: '+str(cur_class1error*100.0/sum(trainCleaned.class_exp.values))+'%') # note: this number is percentage
#model in 2 layers, 0~100, 100+
logging.warning('start regression modeling')
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
interaction = []
alpha = 0.001
for i in featureCombo:
    newFeature = i+'X'+'predict_class_exp'
    trainCleaned[newFeature] = trainCleaned[i]*trainCleaned['predict_class_exp']
    testData[newFeature] = testData[i]*testData['predict_class_exp']
    interaction.append(newFeature)
featureCombo.extend(interaction)
featureCombo.append('predict_class_exp')

model = linear_model.Lasso(alpha = alpha,max_iter=10000)
model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
z = pd.DataFrame(model.predict(testData[featureCombo]),index=testData.index)
logging.warning('features: '+str(featureCombo))
logging.warning('coef: '+str(model.coef_))
z.columns = ['Expected']
z[z<0] = 0 #remove negative prediction
z.to_csv(fn_out)
logging.warning('Prediction done')
logging.warning('Gzipping...')
os.system('gzip -f '+fn_out)
logging.warning('Output written to '+fn_out+'.gz')
