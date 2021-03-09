# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:41:48 2021

@author: andre
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
import numpy as np

#%% data 2 - Til øvelserne..

x = np.linspace(-10,10,1000)
y = np.sinc(x)


fig = plt.figure(dpi=400)
plt.plot(x,y, '.')
plt.title("Data")

x = x.reshape(-1,1) # Scikit-algoritmer kræver (:,1)-format

#%%
# =============================================================================
#  opg a
# =============================================================================
mlp = MLPRegressor(activation = 'tanh', # Aktiveringsfunktionen 
                   hidden_layer_sizes = 2, # Tuple (antal layers, antal neuroner). Antal layers defaulter til 3, hvor ét er output, altså 2 hidden layers. 
                   alpha = 1e-5, 
                   solver = 'lbfgs', # Valg af solver, her quasi-Newton solver
                   max_iter=1000,
                   verbose = True)
mlp.fit(x,y)

fig1 = plt.figure(dpi=400)
plt.plot(x,y)
plt.plot(x, mlp.predict(x), 'rx', ms=1)
plt.title("mlp with 2 hidden neurons, so 2 terms in equation")

#Printer fundne koefficienter. De kommer i rækkefølge, så W1(1) er første værdi. og W1(2) er første værdi i anden array.
#Bias leddene svarer til undervisers 0X-værdier. Den tredje er den overordnede bias, som er tredje hidden neuron.
co = mlp.coefs_
bias = mlp.intercepts_
print(mlp.coefs_) # w-parametre
print(mlp.intercepts_) # = bias led

#%% 
# =============================================================================
# opg b
# =============================================================================
#Tegning, se billede GrafikL06.png

#%% 
# =============================================================================
# opg c
# =============================================================================
#The expression is on the form: 
#y = (W_1^(2)*tanh(W_1^(1)*x+W_{01}^(1))+W_2^(2)*tanh(W_2^(1)*x+W_{02}^(1))+W_0^(2))
print('y=({:.3})*tanh(({:.3})*x+({:.3}))+({:.3})*tanh(({:.3})*x+({:.3})+({:.3})'.format(co[1][0][0],co[0][0][0],bias[0][0],co[1][1][0],co[0][0][1],bias[0][1],bias[1][0]))

#%%
# =============================================================================
#  opg d
# =============================================================================
fig2 = plt.figure(dpi=400)
#Laver funktionen og udvælger de rigtige værdier ift. ovenstående udtryk som kommentar i c
y_calc = (co[1][0][0]*np.tanh(co[0][0][0]*x+bias[0][0])+co[1][1][0]*np.tanh(co[0][0][1]*x+bias[0][1]+bias[1][0]))
plt.plot(x,y_calc)
plt.title('Function = \n({:.3})*tanh(({:.3})*x+({:.3}))+({:.3})*tanh(({:.3})*x+({:.3})+({:.3})'.format(co[1][0][0],co[0][0][0],bias[0][0],co[1][1][0],co[0][0][1],bias[0][1],bias[1][0]))

#%%
# =============================================================================
#  opg e
# =============================================================================
fig3 = plt.figure(dpi=400)
y_calc1 = (co[1][0][0]*np.tanh(co[0][0][0]*x+bias[0][0]))
plt.plot(x,y_calc1,label='1st part')
#plt.title('({:.3})*tanh(({:.3})*x+({:.3}))'.format(co[1][0][0],co[0][0][0],bias[0][0]))

y_calc2 = (co[1][1][0]*np.tanh(co[0][0][1]*x+bias[0][1]))
plt.plot(x,y_calc2, label='2nd part')
#plt.title('({:.3})*tanh(({:.3})*x+({:.3}))'.format(co[1][1][0],co[0][0][1],bias[0][1]))
plt.legend()
plt.title('First and second part of the function')

#%%
# =============================================================================
#  opg f
# =============================================================================
mlp = MLPRegressor(activation = 'tanh', # Aktiveringsfunktionen 
                   hidden_layer_sizes = 5, # Tuple (antal layers, antal neuroner). Antal layers defaulter til 3, hvor ét er output, altså 2 hidden layers. 
                   #Her to hidden layers med 5 neuroner og heraf led i funktionen, eller rettere flere hidden neurons. 
                   alpha = 1e-5, 
                   solver = 'lbfgs', # Valg af solver, her quasi-Newton solver
                   max_iter=1000,
                   verbose = True)
mlp.fit(x,y)

fig4 = plt.figure(dpi=400)
plt.plot(x,y)
plt.plot(x, mlp.predict(x), 'rx', ms=1)
plt.title("mlp with 5 hidden neurons, so 5 terms in equation")
#Printer fundne koefficienter. De kommer i rækkefølge, så W1(1) er første værdi. og W1(2) er første værdi i anden array.
#Bias leddene svarer til undervisers 0X-værdier. Den tredje er den overordnede bias, som er tredje hidden neuron.
co = mlp.coefs_
bias = mlp.intercepts_
print(mlp.coefs_) # w-parametre
print(mlp.intercepts_) # = bias led

#%%
# =============================================================================
# opg g
# =============================================================================
mlp = MLPRegressor(activation = 'tanh', # Aktiveringsfunktionen 
                   hidden_layer_sizes = 2, # Tuple (antal layers, antal neuroner). Antal layers defaulter til 3, hvor ét er output, altså 2 hidden layers. 
                   #Her to hidden layers med 5 neuroner og heraf led i funktionen, eller rettere flere hidden neurons. 
                   alpha = 1e-1, 
                   solver = 'lbfgs', # Valg af solver, her quasi-Newton solver
                   max_iter=1000,
                   verbose = True)
mlp.fit(x,y)

fig4 = plt.figure(dpi=400)
plt.plot(x,y)
plt.plot(x, mlp.predict(x), 'rx', ms=1)
plt.title("mlp with 5 hidden neurons, so 5 terms in equation")
#Printer fundne koefficienter. De kommer i rækkefølge, så W1(1) er første værdi. og W1(2) er første værdi i anden array.
#Bias leddene svarer til undervisers 0X-værdier. Den tredje er den overordnede bias, som er tredje hidden neuron.
co = mlp.coefs_
bias = mlp.intercepts_
print(mlp.coefs_) # w-parametre
print(mlp.intercepts_) # = bias led

# =============================================================================
# Svar
# =============================================================================
#Det lader til at den udjævner den, så der ikke er helt så meget svung og helt så stort et pres for at få den til at passe på alle kurverne
