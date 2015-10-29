import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

#reading in training data
os.chdir('/Users/alex/Dropbox/school/CSCI-567_machine-learning/Kaggle')
fn_train='train_100000.csv'
df=pd.read_csv(fn_train,sep=',')
n=len(list(df.columns.values))

#overview of outcome-variable correlation
for j in range(1,n-1):
    data=df[np.isfinite(df.iloc[:,j])]
    fig = plt.figure()
    plt.scatter(data.iloc[:,j], data['Expected'],s=1)
    plt.savefig(list(data.columns.values)[j]+'.png')

hist,bin_edges=np.histogram(df_retain['Expected'],50)
plt.bar(bin_edges[0:50],hist,width = 10)
plt.savefig('hist.png') 

for j in range(1,n-1):
    data=df[np.isfinite(df.iloc[:,j])]
    MIN=data.min()[j]
    MAX=data.max()[j]
    gap=(MAX-MIN)/10
    RANGE=arange(MIN,MAX,gap) 
    
#remove data with 'Ref'=0
df_retain=df[np.isfinite(df['Ref'])]
df_retain.to_csv('train_100K_Ref-valid.csv',sep=',')

#substituting missing data with mean
for j in range(1,n-1):
    print(j)
    x=df_retain.iloc[:,j]
    current_mean=x.mean()    
    x[np.isnan(x)]=current_mean
df_retain.to_csv('train_100K_Ref-valid_subst.csv',sep=',')

#linear regression
from sklearn import linear_model
clf = linear_model.LinearRegression()
varmat=df_retain.iloc[:,10].as_matrix()
varmat=varmat.reshape((len(df_retain),1))
outcome=df_retain.iloc[:,n-1]
clf.fit(varmat,outcome)
clf.coef_

train_result=np.dot(varmat,clf.coef_)
train_result.mean()




