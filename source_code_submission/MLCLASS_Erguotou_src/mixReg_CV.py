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
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV
cols = trainCleaned.columns[trainCleaned.columns!='Expected']
for i in range(len(cols)):
    for j in range(i,len(cols)):
        newFeature = str(cols[i]+'X'+cols[j])
        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
#featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])
for l in ['hinge', 'log', 'modified_huber', 'squared_hinge','perceptron']:
    for i in [0.1,0.01,0.001,0001,1e-5]:
        print('alpha '+str(i)+' loss '+l)
        model = linear_model.SGDClassifier(loss=l,class_weight='balanced',alpha=i,penalty='l1',n_iter=10)
        model.fit(trainCleaned[featureCombo],trainCleaned['class_exp'])
        trainCleaned['predict_class_exp'] = model.predict(trainCleaned[featureCombo])
        logging.warning('total class_exp=1:'+str(sum(trainCleaned.class_exp.values)))
        logging.warning('error rate:'+str(float(sum(trainCleaned.class_exp != trainCleaned.predict_class_exp))/trainCleaned.shape[0]))
        logging.warning('total class_exp=1 predicted wrong:'+str(sum(trainCleaned.class_exp[trainCleaned.class_exp != trainCleaned.predict_class_exp].values)))

#losshinge mean error rate 50-fold CV: 29.46%
#losslog mean error rate 50-fold CV: 22.92%
#lossmodified_huber mean error rate 50-fold CV: 23.86%
#losssquared_hinge mean error rate 50-fold CV: 18.88%
#lossperceptron mean error rate 50-fold CV: 26.14%

#model in 2 layers, 0~100, 100+
featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
interaction = []
for i in featureCombo:
    newFeature = i+'X'+'predict_class_exp'
    trainCleaned[newFeature] = trainCleaned[i]*trainCleaned['predict_class_exp']
    interaction.append(newFeature)
featureCombo.extend(interaction)
featureCombo.append('predict_class_exp')

def runModel(i,featureCombo):
    mae = np.array([])   
    logging.warning('try alpha = %s' % i)
    coef = None
    for ktrain,ktest in kf:
	predictor = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,featureCombo)]
	predictor2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,featureCombo)]
	out = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'Expected')]
	out2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'Expected')]
	predictor = predictor[out.Expected < 10]
	out = out[out.Expected < 10]
        model = linear_model.Lasso(alpha = i,max_iter=100000,warm_start=True)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    logging.warning('coef: %s' % coef)
    logging.warning('aggByMean, average 50-fold MAE for alpha %s feature %s: %s ' % (i,featureCombo,mae.mean()))

logging.warning('start regression modeling')
kf = KFold(trainCleaned.shape[0],n_folds=50,shuffle=False)
for i in [1e-10,1e-8,1e-6,1e-4,1e-3]:
    runModel(i,featureCombo)
