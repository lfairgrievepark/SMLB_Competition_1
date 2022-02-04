#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:39:51 2021

@author: lfairgrievepark12
"""

1
2
3
4
# import our packages
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score



# Turning car types and fuel types into integer classifications for model
# to interpret
data=pd.read_csv('trainingData.csv')
def replace_cartype(cartype):
    try:
        types1= ['van','regcar','truck','sportuv','sportcar','stwagon']
        return types1.index(cartype)
    except ValueError:
        return cartype
    
def replace_fueltype(fueltype):
    try:
        types1 = ['cng','methanol','electric','gasoline']
        return types1.index(fueltype)
    except ValueError:
        return fueltype
    

    
for name in ['type1','type2','type3','type4','type5','type6']:
    data[name] = data[name].apply(replace_cartype)
for name in ['fuel1','fuel2','fuel3','fuel4','fuel5','fuel6']:
    data[name] = data[name].apply(replace_fueltype)

# Turning y value (car selected) into integers as well
data.replace(['choice1','choice2','choice3','choice4','choice5','choice6'],[1,2,3,4,5,6],inplace=True)

data=data.drop(['id'],axis=1)
y = data[['choice']]
X = data.drop(['choice'],axis=1)

#One hot encoding y data
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# Trying to maximize training data, probably need  a better split    
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.028,random_state=93)

# Set early stopping parameters and learning rate reduction parameters
earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='auto')
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.04, patience=7, verbose=1, epsilon=1e-4, mode='auto')

# define the keras model
model = Sequential()
model.add(Dense(100, input_dim=69, activation='elu'))
model.add(Dense(60, activation='selu'))
model.add(Dense(30, activation='elu'))
model.add(Dense(6, activation='softmax'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(Xtrain, ytrain, epochs=1000, batch_size=50, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

# evaluate the keras model
ypred = model.predict(Xtest)

#Converting predictions to label
pred = list()
for i in range(len(ypred)):
    pred.append(np.argmax(ypred[i]))
print(pred)

#Converting one hot encoded test label to label
test = list()
for i in range(len(ytest)):
    test.append(np.argmax(ytest[i]))
print(test)

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

    


