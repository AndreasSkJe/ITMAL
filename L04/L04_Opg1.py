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

fig2 = plt.figure(dpi=400)
plt.scatter(X[:,1], y)
plt.title("Disease progression vs normalised Sex")
#Højere bund ved højre gruppe af værdier, højere top. 

fig3 = plt.figure(dpi=400)
plt.scatter(X[:,2], y)
plt.title("Disease progression vs normalised BMI")
#Ligner en form for regression med h'jere BMi der giver højere værdi for hvor fremskudt sygdom er

fig4 = plt.figure(dpi=400)
plt.scatter(X[:,3], y)
plt.title("Disease progression vs normalised Average Blood Pressure")
# Også en tendens til højere værdier som funktion af blodtryk.

#%%
# =============================================================================
# Opgave 1b
# =============================================================================
model = linear_model.LinearRegression()
X_BMI = X[:,3].reshape(1,-1).transpose()
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
model.fit(X,y)

y_pred = model.predict(X)

RMSE_e = metrics.mean_squared_error(y,y_pred, squared=False)  #Returnerer RMSE hvis squared er false
print("The RMSE is: "+ str(RMSE_e))


