# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:50:53 2019

@author: user
"""

import pandas as pd
data=pd.read_excel('Reliability modelling.xlsx')


correlation=data.corr()

import seaborn as sns
sns.heatmap(correlation)

#sns.pairplot(data,kind="scatter")

x=data.iloc[:,[3,4,5,6]].values
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
data2=data1.drop(['X2'],axis=1)
data3=data2.drop(['X9'],axis=1)
data4=data3.drop(['X8'],axis=1)
data5=data4.drop(['X3'],axis=1)
data6=data5.drop(['X1'],axis=1)
vif_cal(data6,'Y1')
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
#0.53

import numpy as np
#new metrics
# finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test))

# finding the RMSE
from sklearn.metrics import mean_squared_error
base_RMSE =(mean_squared_error(y_test,base_pred))**0.5                            
print(base_RMSE)
#0.64


datam=pd.read_excel('Reliability modelling.xlsx',sheet_name='X_Test')
y_pred1=regress.predict(datam)







#Model Score  - Sk learn Library is must

from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
x=SC.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.linear_model import LinearRegression
Reg=LinearRegression()
Reg.fit(x,y)
Reg.score(x_test,y_test)

y_pred=Reg.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
print(sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
#0.407
#0.602








