# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:40:32 2021

@author: andre
"""
from sklearn import *
import matplotlib.pyplot as plt
import numpy as np

[X,y] = datasets.load_diabetes(return_X_y=True, as_frame=False)

#Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements 

#%%
# =============================================================================
# Opgave 1a
# =============================================================================

fig1 = plt.figure(dpi=400)
plt.scatter(X[:,0], y)
plt.title("Disease progression vs normalised Age")
plt.xlabel("Normalised Age")
plt.ylabel("Disease Progression")

fig2 = plt.figure(dpi=400)
plt.scatter(X[:,1], y)
plt.title("Disease progression vs normalised Sex")
plt.xlabel("Normalised Sex")
plt.ylabel("Disease Progression")

fig = plt.figure(dpi=400)
G = np.asarray([X[:,1],y])
oneG = []
otherG = []
for i in range(len(y)):
    if G[:,i][0] == 0.0506801187398187:
        oneG.append(G[:,i])
    elif G[:,i][0] == -0.044641636506989:
       otherG.append(G[:,i])
     
plt.boxplot(np.asarray([oneG, otherG]))
plt.title("Disease progression vs normalised Sex")
plt.xlabel("Normalised Sex")
plt.ylabel("Disease Progression")
#Højere bund ved højre gruppe af værdier, højere top. 

fig4 = plt.figure(dpi=400)
plt.scatter(X[:,3], y)
plt.title("Disease progression vs normalised Average Blood Pressure")
plt.xlabel("Normalised Average Blood Pressure")
plt.ylabel("Disease Progression")

# Også en tendens til højere værdier som funktion af blodtryk.

fig3 = plt.figure(dpi=400)
plt.scatter(X[:,2], y)
plt.title("Disease progression vs normalised BMI")
plt.xlabel("Normalised BMI")
plt.ylabel("Disease Progression")

#Ligner en form for regression med h'jere BMi der giver højere værdi for hvor fremskudt sygdom er



#%%
# =============================================================================
# Opgave 1b
# =============================================================================
model = linear_model.LinearRegression()
X_BMI = X[:,2].reshape(1,-1).transpose()
model.fit(X_BMI,y)

y_pred = model.predict(X_BMI)

RMSE_b = metrics.mean_squared_error(y,y_pred, squared=False)  #Returnerer RMSE hvis squared er false
print("The RMSE is: "+ str(RMSE_b))

#%%
# =============================================================================
# Opgave 1c
# =============================================================================

#Coefficienter
theta_1 = model.coef_
theta_0 =  model.intercept_

fig3 = fig3
plt.plot(X_BMI,y_pred,color='r')
#Ser ud til at plotte fint. 

#%%
# =============================================================================
# Opgave 1d
# =============================================================================

residuals = (y - y_pred)
fig5 = plt.figure(dpi=400)
plt.hist(residuals, bins=20)
plt.title("Histogram for residuals of fit for Diabetes vs BMI")
plt.xlabel("Residual")
plt.ylabel("Number of residuals")

#%% 
# =============================================================================
# Opgave 1e
# =============================================================================
X_first4 = X[:,0:4]

model = linear_model.LinearRegression()
model.fit(X_first4,y)

y_pred = model.predict(X_first4)

RMSE_e = metrics.mean_squared_error(y,y_pred, squared=False)  #Returnerer RMSE hvis squared er false
print("The RMSE is: "+ str(RMSE_e))
#Ny fejl er mindre. 

#%%
# =============================================================================
# Opgave 1f
# =============================================================================
X_all = X

model = linear_model.LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

RMSE_e = metrics.mean_squared_error(y,y_pred, squared=False)  #Returnerer RMSE hvis squared er false
print("The RMSE is: "+ str(RMSE_e))
#Ny RMSE endnu bedre. 

#%% 
# =============================================================================
# Opgave 1g
# =============================================================================

vals70pct = int(len(y)*0.7)
X_train, X_test, y_train, y_test = X[:vals70pct], X[vals70pct:], y[:vals70pct], y[vals70pct:]

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

RMSE_g = metrics.mean_squared_error(y_test,y_pred, squared=False)
print("The RMSE is: "+ str(RMSE_g))

#Trying with the data in other order, so the first 30% are test, and last 70% are training.
vals30pct = int(len(y)*0.3)
X_train, X_test, y_train, y_test = X[vals30pct:], X[:vals30pct], y[vals30pct:], y[:vals30pct]

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

RMSE_g = metrics.mean_squared_error(y_test,y_pred, squared=False)  
print("The RMSE is: "+ str(RMSE_g))

# As I get a smaller error for the second attempt on the split test and train dataset, there's no evidence of an overfit. 

