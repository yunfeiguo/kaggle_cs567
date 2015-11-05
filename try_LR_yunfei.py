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
testCleaned = testData.groupby(testData.index).filter(lambda x: sum(np.isfinite(x['Ref'])) > 1)
#fill remaining NaNs with column mean
#alternatively, try group mean
trainCleaned.fillna(cleaned.mean().to_dict(),inplace=True)
testCleaned.fillna(cleaned.mean().to_dict(),inplace=True)
#make sure there is no NaN any more
sum(np.isfinite(trainCleaned)==False) #return all zeros
sum(np.isfinite(testCleaned)==False)

#scatter plot
manyScatter(cleaned,range(22),22)

#do some transformation, then scatter plot
aggBySum = cleaned.groupby(cleaned.index).agg(np.sum)
testNoAgg = test.copy()
test = test.groupby(test.index).agg(np.sum)
manyScatter(aggBySum,range(aggBySum.shape[1]-1),aggBySum.shape[1]-1)
#try a series of trigonometric transformations
y = aggBySum.groupby('radardist_km').agg(np.sum)
y = y[y['Expected']<10000]
manyScatter(y,range(y.shape[1]-1),y.shape[1]-1)


#overview of outcome-variable correlation
for j in range(1,n-1):
    fig = plt.figure()
    plt.scatter(cleaned.iloc[:,j], cleaned['Expected'],s=1)
    plt.ylabel('Expected')
    plt.xlabel(cleaned.columns[j])
    plt.savefig(cleaned.columns[j]+'.png')

cleaned.groupby(cleaned.index).transform(lambda x: x.mean)
plt.hist(df['Expected'].as_matrix(),bins=100)
plt.savefig('hist.png')   
        

#data preparation
varmat=df_retain.iloc[:,2:n-1].as_matrix()
#varmat=varmat.reshape((len(df_retain),1))
outcome=df_retain.iloc[:,n-1].as_matrix()

#linear regression
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(varmat,outcome)
clf.coef_

#linear ridge regression
from sklearn import linear_model
clf = linear_model.Ridge (alpha = .5)
clf.fit(varmat,outcome)

#lasso
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(varmat,outcome)

#Bayesian Ridge
clf = linear_model.BayesianRidge()
clf.fit(varmat,outcome)

#Analyze error
train_result=np.dot(varmat,clf.coef_)
MAE=sum(abs(train_result-outcome))/len(df_retain)

fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(y['Ref'],y['Expected'],y.index)
plt.show()
plt.xlabel('Ref')
plt.ylabel('Expected')

fig = pylab.figure()
ax = Axes3D(fig)
data = aggBySum[aggBySum<1000]
data = data.loc[data['radardist_km']<=10,:]
ax.scatter(data['Ref'],data['Expected'],data['radardist_km'])
plt.show()
plt.xlabel('Ref')
plt.ylabel('Expected')


plt.figure();plt.scatter(aggBySum['Ref']*aggBySum['radardist_km'],aggBySum['Expected']);
small = aggBySum.loc[aggBySum['Expected']<1000,:]
plt.figure();plt.scatter(small['Ref']*small['radardist_km'],small['Expected']);

#test interaction term Ref*radardist_km
clf = linear_model.LinearRegression()
refAndDist = aggBySum.loc[:,['Ref','radardist_km']]
clf.fit(refAndDist,aggBySum['Expected'])
getMAE(np.dot(test.loc[:,['Ref','radardist_km']],clf.coef_),test['Expected'])
#now add interaction term
clf = linear_model.LinearRegression()
refAndDist = pd.DataFrame({'Ref':aggBySum['Ref'],'radardist_km':aggBySum['radardist_km'],'interaction':aggBySum['Ref']*aggBySum['radardist_km']})
clf.fit(refAndDist,aggBySum['Expected'])
testRefAndDist = pd.DataFrame({'Ref':test['Ref'],'radardist_km':test['radardist_km'],'interaction':test['Ref']*test['radardist_km']})
getMAE(np.dot(testRefAndDist,clf.coef_),test['Expected'])


#also, write a function for generating benchmark MAE
est = testNoAgg.groupby(testNoAgg.index).apply(getBenchMark)
getMAE(est,testNoAgg['Expected'])

#split the data by distance categories (perhaps use 10km bins)
#then do regression within each bin
#see if interaction effects exist

x = trainCleaned.groupby(trainCleaned.index).agg(np.mean)
#x = x[['Ref','radardist_km','RhoHV','Zdr','Kdp','Expected']]
#x = x[['Ref','Expected']]
x = x[['radardist_km','Expected']]
#x = x[['RhoHV','Expected']]
#x = x[['Zdr','Expected']]
#x = x[['Kdp','Expected']]
#linear elastic net regression
for i in frange(0,1,0.1):
    print('alpha:'+str(i))
    #xElasticNet = linear_model.ElasticNet (alpha = i)
    xElasticNet = linear_model.Ridge (alpha = i)
    xElasticNet.fit(x.iloc[:,0:(x.shape[1]-1)],x['Expected'])
    getMAE(np.dot(x.iloc[:,0:(x.shape[1]-1)],xElasticNet.coef_),x['Expected'])

manyScatter(x,range(23),4)


x.groupby('radardist_km').apply(
