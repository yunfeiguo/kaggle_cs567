#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import random
from yg_utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
pd.options.mode.chained_assignment = None 
#os.chdir("/auto/rmscr/shared_resources/qiuyuguo/kaggle_cs567/scripts_to_submit")
#os.chdir("/Users/alex/Dropbox/school/classes_usc/CSCI-567_machine-learning/Kaggle/scripts_to_submit")

#data=pd.read_csv(fn_train,sep=',',index_col='Id')
#smalltrain=data.iloc[np.random.random_integers(0,data.shape[0]-1,int(data.shape[0]*0.1))]
#smalltrain.to_csv('./data/smalltrain.csv')

fn_train='data/aggmean_cleaned_smalltrain_reducedFeature.csv'
all=pd.read_csv(fn_train,sep=',',index_col='Id')

#fn_small='data/smalltrain2.csv'
fn_small='data/smalltrain2.csv'
small=pd.read_csv(fn_small,sep=',',index_col='Id')

classification_featureCombo = all.columns[all.columns!='Expected']

# Linear Ridge ###############################################################

# Linear Ridge parameter tuning
from sklearn import linear_model
result=[]
kf = KFold(small.shape[0],n_folds=10,shuffle=False)
for alpha in [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100]:
    kresult=np.array([])
    for ktrain,ktest in kf:
        
        ktrainData = small.iloc[ktrain,]
        ktestData = small.iloc[ktest,]
        #segregation
        ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
        ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
        ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
        
        #train segregated model
        model=linear_model.Ridge (alpha = alpha)
        model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
        
        #compute MAE using mixture model
        pred=model.predict(ktestData[classification_featureCombo])
        mae=getMAE(pred,ktestData['Expected'])
        kresult=np.append(kresult,mae)
    
    print('alpha='+str(alpha)+',MAE='+str(kresult.mean()))
    result.append('alpha='+str(alpha)+',MAE='+str(kresult.mean()))

out=open('data/LinearRidge_cv_result.txt','w')
for i in range(len(result)):
    out.write(result[i]+'\n')
out.close()

# Linear Ridge CV on full data
kf = KFold(all.shape[0],n_folds=10,shuffle=False)
alpha=1e-6
kresult=np.array([])
for ktrain,ktest in kf:
    
    ktrainData = all.iloc[ktrain,]
    ktestData = all.iloc[ktest,]
    #segregation
    ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
    ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
    ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
    
    #train segregated model
    model=linear_model.Ridge (alpha = alpha)
    model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
    
    #compute MAE using mixture model
    pred=model.predict(ktestData[classification_featureCombo])
    mae=getMAE(pred,ktestData['Expected'])
    kresult=np.append(kresult,mae)

print('alpha='+str(alpha)+',MAE='+str(kresult.mean()))

# Bayesian Ridge ###############################################################

# Bayesian Ridge parameter tuning
from sklearn import linear_model
result=[]
kf = KFold(small.shape[0],n_folds=10,shuffle=False)
for alpha in [1e-8,1e-6,1e-4,1e-2,1]:
    for lamda in [1e-8,1e-6,1e-4,1e-2,1]:
        kresult=np.array([])
        for ktrain,ktest in kf:
            
            ktrainData = small.iloc[ktrain,]
            ktestData = small.iloc[ktest,]
            #segregation
            ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
            ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
            ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
            
            #train segregated model
            model=linear_model.BayesianRidge(alpha_1=alpha,alpha_2=alpha,lambda_1=lamda,lambda_2=lamda)
            model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
            
            #compute MAE using mixture model
            pred=model.predict(ktestData[classification_featureCombo])
            mae=getMAE(pred,ktestData['Expected'])
            kresult=np.append(kresult,mae)
        
        print('alpha='+str(alpha)+',lambda='+str(lamda)+',MAE='+str(kresult.mean()))
        result.append('alpha='+str(alpha)+',lambda='+str(lamda)+',MAE='+str(kresult.mean()))

out=open('data/BayesianRidge_cv_result.txt','w')
for i in range(len(result)):
    out.write(result[i]+'\n')
out.close()

# Bayasian Ridge on full data
kf = KFold(all.shape[0],n_folds=50,shuffle=False)
alpha=1e-6
lamda=1e-6
kresult=np.array([])
for ktrain,ktest in kf:
            
    ktrainData = all.iloc[ktrain,]
    ktestData = all.iloc[ktest,]
    #segregation
    ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
    ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
    ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
            
    #train segregated model
    model=linear_model.BayesianRidge(alpha_1=alpha,alpha_2=alpha,lambda_1=lamda,lambda_2=lamda)
    model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
            
    #compute MAE using mixture model
    pred=model.predict(ktestData[classification_featureCombo])
    mae=getMAE(pred,ktestData['Expected'])
    kresult=np.append(kresult,mae)
        
print('alpha='+str(alpha)+',lambda='+str(lamda)+',MAE='+str(kresult.mean()))

# Random Forest ###############################################################

# Random Forest parameters tuning

from sklearn.ensemble import RandomForestRegressor
result=[]
kf = KFold(small.shape[0],n_folds=10,shuffle=False)
#for K in [10,100,200,500,1000]:
for K in [10,20]:
    for md in [1,3]:
        kresult=np.array([])
        for ktrain,ktest in kf:
            
            ktrainData = small.iloc[ktrain,]
            ktestData = small.iloc[ktest,]
            #segregation
            ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
            ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
            ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
            
            #train segregated model
            model = RandomForestRegressor(n_estimators=K,max_depth=md)
            model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
            
            #compute MAE using mixture model
            pred=model.predict(ktestData[classification_featureCombo])
            mae=getMAE(pred,ktestData['Expected'])
            kresult=np.append(kresult,mae)
        
        print('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))
        result.append('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))

out=open('data/RandomForest_cv_result.txt','w')
for i in range(len(result)):
    out.write(result[i]+'\n')
out.close()

# Random Forest cv on full data

kf = KFold(all.shape[0],n_folds=10,shuffle=False)
K=5
md=2
kresult=np.array([])
for ktrain,ktest in kf:
    
    ktrainData = all.iloc[ktrain,]
    ktestData = all.iloc[ktest,]
    #segregation
    ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
    ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
    ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
    
    #train segregated model
    model = RandomForestRegressor(n_estimators=K,max_depth=md)
    model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
    
    #compute MAE using mixture model
    pred=model.predict(ktestData[classification_featureCombo])
    mae=getMAE(pred,ktestData['Expected'])
    kresult=np.append(kresult,mae)

print('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))

# GBR ###############################################################

# GBR parameter tuning
from sklearn.ensemble import GradientBoostingRegressor
result=[]
kf = KFold(small.shape[0],n_folds=10,shuffle=False)
#for K in [10,100,200,500,1000]:
for K in [1,2]:
    #for md in [1,3,5,10]:
    for md in [1,3,5]:
        kresult=np.array([])
        for ktrain,ktest in kf:
            
            ktrainData = small.iloc[ktrain,]
            ktestData = small.iloc[ktest,]
            #segregation
            ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
            ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
            ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
            
            #train segregated model
            from sklearn import linear_model
            model=GradientBoostingRegressor(n_estimators=K, learning_rate=0.1,max_depth=md, random_state=0, loss='ls')
            model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
            
            #compute MAE using mixture model
            pred=model.predict(ktestData[classification_featureCombo])
            mae=getMAE(pred,ktestData['Expected'])
            kresult=np.append(kresult,mae)
        
        print('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))
        result.append('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))

out=open('data/gbr_cv_result.txt','w')
for i in range(len(result)):
    out.write(result[i]+'\n')
out.close()

# GBR CV all full data
kf = KFold(all.shape[0],n_folds=10,shuffle=False)
K=10
md=5
kresult=np.array([])
for ktrain,ktest in kf:
    
    ktrainData = all.iloc[ktrain,]
    ktestData = all.iloc[ktest,]
    #segregation
    ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
    ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
    ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
    
    #train segregated model
    model=GradientBoostingRegressor(n_estimators=K, learning_rate=0.1,max_depth=md, random_state=0, loss='ls')
    model.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
    
    #compute MAE using mixture model
    pred=model.predict(ktestData[classification_featureCombo])
    mae=getMAE(pred,ktestData['Expected'])
    kresult=np.append(kresult,mae)

print('K='+str(K)+',md='+str(md)+',MAE='+str(kresult.mean()))

# RANSAC ###############################################################

# RANSAC CV on full data
from sklearn import linear_model
kresult=np.array([])
kf = KFold(small.shape[0],n_folds=10,shuffle=False)
for ktrain,ktest in kf:
    
    ktrainData = small.iloc[ktrain,]
    ktestData = small.iloc[ktest,]
    #segregation
    ktrainData['class_exp']=pd.cut(ktrainData['Expected'],[0,100,1e6],labels=[0,1])
    ktrainData1=ktrainData.loc[ktrainData['class_exp']==0]
    ktrainData2=ktrainData.loc[ktrainData['class_exp']==1]
    
    #train segregated model
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(ktrainData1[classification_featureCombo],ktrainData1['Expected'])
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    #compute MAE using mixture model
    pred=model_ransac.predict(ktestData[classification_featureCombo])
    mae=getMAE(pred,ktestData['Expected'])
    kresult=np.append(kresult,mae)

print('MAE='+str(kresult.mean()))

out=open('data/RANSAC_cv_result.txt','w')
out.write('MAE='+str(kresult.mean())+'\n')
