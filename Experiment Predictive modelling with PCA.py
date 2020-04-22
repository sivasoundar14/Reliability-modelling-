# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:00:38 2019

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_excel('Reliability modelling.xlsx')

m=data.iloc[:,:10].values
n=data.iloc[:,10].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
m=sc.fit_transform(m)
#Standard Scaler returns a float array therefore convert into dataframe mandatory

m=pd.DataFrame(m)
from sklearn.decomposition import PCA
pca=PCA()
m=pca.fit_transform(m)     #Fits the components after rotational transform

pca.components_[0]

res=pca.explained_variance_ratio_*100
mar=np.cumsum(pca.explained_variance_ratio_*100)

scores=pd.Series(pca.components_[0])      
scores.abs().sort_values(ascending=False)   # # Displays index Absolute Eigen values

var=pca.components_[0]
##Scree Plot
plt.bar(x=range(1,len(var)+1),height=mar)
plt.xlabel('Factors')
plt.ylabel('% Explained Variance')

#This implies Feature 1,3,7,8,6,9 contribute to about 96% of the variances 
#Build model based on these parameters 

correlation=data.corr()

import seaborn as sns
sns.heatmap(correlation)

x=data.iloc[:,[0,2,6,7,5,8]].values
y=data.iloc[:,10].values

import statsmodels.formula.api as sm
def vif_cal(input_data,dependent_col):
    x_var=input_data.drop([dependent_col],axis=1)
    xvar_names=x_var.columns
    for i in range(0,len(xvar_names)):
        y=x_var[xvar_names[i]]
        x=x_var[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols("y~x",x_var).fit().rsquared
        vif=round(1/(1-rsq),2)
        print(xvar_names[i],"VIF:",vif)
data1=data.drop(['X10'],axis=1)
data2=data1.drop(['X4'],axis=1)
data3=data2.drop(['X5'],axis=1)
data4=data3.drop(['X2'],axis=1)

vif_cal(data4,'Y1')
#Drop X1 find Rsquare Value 


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


import statsmodels.formula.api as sm
regress=sm.OLS(y_train,x_train).fit()
regress.summary()

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
y_pred=regress.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_pred)))
#0.53 (Model 1 without PCA)
#0.59 (Model 2 With PCA)














