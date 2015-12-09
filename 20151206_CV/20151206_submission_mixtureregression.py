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

parser = argparse.ArgumentParser()
parser.add_argument('-k',nargs='+',required=True)
parser.add_argument('-p',nargs='+',required=True)
parser.add_argument('-a',nargs='+',required=True)
parser.add_argument('-w',nargs='+',required=True)
args = parser.parse_args()

#reading in training data
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_1e7.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/test_for_1e7.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/train_100000.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/smalltest.v2.csv'
#fn_out='/home/yunfeiguo/projects/kaggle_cs567/results/20151206_mixture_knn_k5_p2_alpha1e-3.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltrain.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.v2.csv'
#fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train_70k.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_smalltest.v2.csv'
fn_train='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_train.v2.csv'
#fn_test='/home/yunfeiguo/projects/kaggle_cs567/data/aggmean_cleaned_test.v2.csv'
trainCleaned=pd.read_csv(fn_train,sep=',',index_col='Id')
#testData = pd.read_csv(fn_test,sep=',',index_col='Id')
logging.warning('reading done')

# classification by logistic and learn parameter by CV
cols = trainCleaned.columns[trainCleaned.columns!='Expected']
#for i in range(len(cols)):
#    for j in range(i,len(cols)):
#        newFeature = str(cols[i]+'X'+cols[j])
#        trainCleaned[newFeature] = trainCleaned[cols[i]]*trainCleaned[cols[j]]
#        testData[newFeature] = testData[cols[i]]*testData[cols[j]]
classification_featureCombo = trainCleaned.columns[trainCleaned.columns!='Expected']
trainCleaned['class_exp']=pd.cut(trainCleaned['Expected'],[0,100,1e6],labels=[0,1])

from sklearn import neighbors as nb
kf = KFold(trainCleaned.shape[0],n_folds=10,shuffle=False)

#model in 2 layers, 0~100, 100+
#alpha = 0.001
#K = 5
#p = 2
alpha = float(args.a[0])
K = int(args.k[0])
p = int(args.p[0])
weight = args.w[0]

#model = linear_model.Lasso(alpha = alpha,max_iter=10000)
#model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
#z = pd.DataFrame(model.predict(testData[featureCombo]),index=testData.index)
#logging.warning('features: '+str(featureCombo))
#logging.warning('coef: '+str(model.coef_))
#z.columns = ['Expected']
#z[z<0] = 0 #remove negative prediction
#z.to_csv(fn_out)
#logging.warning('Prediction done')
#logging.warning('Gzipping...')
#os.system('gzip -f '+fn_out)
#logging.warning('Output written to '+fn_out+'.gz')

def runModel(i,featureCombo):
    mae = np.array([])   
    error = np.array([])
    c1error = np.array([])
    logging.warning('try alpha = %s' % i)
    coef = None
    for ktrain,ktest in kf:
	predictor = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,classification_featureCombo)]
	predictor2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,classification_featureCombo)]
	classout = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'class_exp')]
	classout2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'class_exp')]
	out = trainCleaned.iloc[ktrain,np.in1d(trainCleaned.columns,'Expected')]
	out2 = trainCleaned.iloc[ktest,np.in1d(trainCleaned.columns,'Expected')]

        #logging.warning('start classification')
        class_model = nb.KNeighborsClassifier(K,weights=weight,p=p)
        class_model.fit(predictor[classification_featureCombo],np.ravel(classout))
        predictor['predict_class_exp'] = class_model.predict(predictor[classification_featureCombo])
        predictor2['predict_class_exp'] = class_model.predict(predictor2[classification_featureCombo])
        #logging.warning(sum(abs(predictor2['predict_class_exp'].values - classout2.values)))
        #logging.warning(type(float(sum(abs(predictor2['predict_class_exp'].values - classout2.values)))))
	#logging.warning(type(predictor2))
	#logging.warning(type(float(predictor2.shape[0])))
        cur_error_percent=float(sum(abs(predictor2['predict_class_exp'].values - classout2.values)))*100.0/float(predictor2.shape[0])
	#logging.warning(classout2.class_exp)
	#logging.warning(type(classout2.class_exp))
	#logging.warning(predictor2['predict_class_exp'] != classout2.class_exp)
	#logging.warning(type(predictor2['predict_class_exp'] != classout2.class_exp))
        cur_class1error=sum(classout2[predictor2['predict_class_exp'] != classout2.class_exp].values)
	error = np.append(error,cur_error_percent)
	c1error = np.append(c1error,cur_class1error)
        #logging.warning('p '+str(p)+' K'+str(K)+' mean error rate : '+str(cur_error_percent)+'%') # note: this number is percentage
        #logging.warning('p '+str(p)+' K'+str(K)+' mean class1 error rate: '+str(cur_class1error*100.0/sum(trainCleaned.class_exp.values))+'%') # note: this number is percentage

	predictor = predictor.drop('predict_class_exp',1)
        featureCombo = ['Zdr', 'radardist_km', 'MarshallPalmer', 'Katsumata', 'Brandes', 'Sachidanazrnic', 'RyzhkovZrnic']
	featureCombo_for_test = featureCombo[:]
        interaction = []
        interaction_for_test = []
        #for i in ['MarshallPalmer']:
        #    newFeature = i+'X'+'predict_class_exp'
        #    predictor[newFeature] = predictor[i]*predictor['predict_class_exp']
        #    predictor2[newFeature] = predictor2[i]*predictor2['predict_class_exp']
        #    interaction.append(newFeature)
        #featureCombo.extend(interaction)
        #featureCombo.append('predict_class_exp')
        #logging.warning('start regression modeling')
	#we need to use the correct classification for training data
        for i in featureCombo:
            newFeature = i+'X'+'class_exp'
            newFeature_for_test = i+'X'+'predict_class_exp'
            predictor[newFeature] = predictor[i]*(classout.class_exp.astype('int'))
            predictor2[newFeature_for_test] = predictor2[i]*predictor2['predict_class_exp']
            interaction.append(newFeature)
            interaction_for_test.append(newFeature_for_test)
        featureCombo.extend(interaction)
        featureCombo.append('class_exp')
        featureCombo_for_test.extend(interaction_for_test)
	featureCombo_for_test.append('predict_class_exp')
        logging.warning('start regression modeling')
	predictor['class_exp'] = classout.class_exp.astype('int')
        model = linear_model.Lasso(alpha = alpha,max_iter=100000,warm_start=True)
	#model = linear_model.SGDRegressor(loss='squared_loss',penalty='l1',alpha=i)
        model.fit(predictor,out)
	mae = np.append(mae,getMAE(model.predict(predictor2),out2.Expected))
	coef = model.coef_
    #logging.warning('coef: %s' % coef)
    #logging.warning('aggByMean, average 10-fold error for K %s p %s alpha %s feature %s: %s ' % (K,p,alpha,featureCombo,error.mean()))
    #logging.warning('aggByMean, average 10-fold class1 error for K %s p %s alpha %s feature %s: %s ' % (K,p,alpha,featureCombo,c1error.mean()/float(trainCleaned.shape[0])))
    logging.warning('aggByMean, average 10-fold MAE for K %s p %s alpha %s feature %s: %s ' % (K,p,alpha,featureCombo,mae.mean()))

logging.warning('start regression modeling')
runModel(alpha,[])

#model = linear_model.Lasso(alpha)
#model.fit(trainCleaned[featureCombo],trainCleaned['Expected'])
#z = xModel.predict(y[featureCombo])
#z[z<0] = 0 #remove negative prediction
#ae = abs(allExp-allEst)
#ExpectedCut = pd.cut(allExp,[0,10,100,1e6])
#aeWithCut = pd.DataFrame({'ae':ae,'ExpectedCut':ExpectedCut})
#aeSum = aeWithCut.groupby(aeWithCut.ExpectedCut).agg(np.sum)
