#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:39:51 2021

@author: lfairgrievepark12
 """


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
from sklearn.preprocessing import OneHotEncoder

def get_results(name_of_test_file):

    # Importing data
    name_of_file=('trainingData.csv')
    train_data=pd.read_csv(name_of_file)
    test_data=pd.read_csv(name_of_test_file)
    
    # Get rid of annoying string data
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
        train_data[name] = train_data[name].apply(replace_cartype)
        test_data[name] = test_data[name].apply(replace_cartype)
    for name in ['fuel1','fuel2','fuel3','fuel4','fuel5','fuel6']:
        train_data[name] = train_data[name].apply(replace_fueltype)
        test_data[name] = test_data[name].apply(replace_fueltype)
        
    
    # Format data
    train_data.replace(['choice1','choice2','choice3','choice4','choice5','choice6'],[1,2,3,4,5,6],inplace=True)
    train_data=train_data.drop(['id'],axis=1)
    y = train_data[['choice']]
    X = train_data.drop(['choice'],axis=1)
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()
    
    test_data=test_data.drop(['id'],axis=1)
    Xtest=test_data
    
    # Split data, because idk if my algorithm is still accurate unless
    # using these specific params
    Xtrain, na1, ytrain, na2 = train_test_split(X,y,test_size=0.028,random_state=93)
    
    # Set up some parameters to allow the keras model to function better
    earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='auto')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.04, patience=7, verbose=1, epsilon=1e-4, mode='auto')
    
    # define and evaluate keras model
    model = Sequential()
    model.add(Dense(100, input_dim=69, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Xtrain, ytrain, epochs=1000, batch_size=50, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    
    ypred = model.predict(Xtest)
    pred = list()
    for i in range(len(ypred)):
        pred.append(np.argmax(ypred[i])+1)
    my_predictions=[]
    for i in range(0,len(pred)):
        my_predictions.append(['id'+str(i),pred[i]])
    return(my_predictions)


    
a = get_results('testingData.csv')
print(a)

