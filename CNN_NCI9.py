# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thursday May 17 23:00:57 2018
This code run on spyder (python2.7) using theano backend
@author: By Ali Alani to apply in CNN model
"""
#%%
#Import the required packages
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import adam
from keras.utils import np_utils
# import the necessary packages for CNN model 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Flatten
import numpy
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import itertools
#%%
root_datase = 'C:\Users\Lenovo\CNN_KERAS_MNIST\Sadegh\Dataset'
NCI9 = np.genfromtxt(root_datase + '/NCI9_Feat.txt', delimiter=",")
X = np.array(NCI9)
NCI9_1 = np.loadtxt(root_datase + '/NCI9_Target.txt', delimiter=",")
Y = np.array(NCI9_1)
#%%
X= (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + 0.001)
# STEP 1: split X and y into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=4)
print("Training and Testing data shapes:")
print("trainX.shape: {}".format(trainX.shape))
print("trainY.shape: {}".format(trainY.shape))
print("testX.shape: {}".format(testX.shape))
print("testY.shape: {}".format(testY.shape))
#%%
seed = 100
numpy.random.seed(seed)
s1 = trainX.shape[0]
s2 = testX.shape[0]
newshape = (s1, 9712, 1)
newshape1= (s2,9712, 1)
trainX = numpy.reshape(trainX, newshape)
testX =numpy.reshape(testX, newshape1)
# one hot encode outputs
trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
print("Training and Testing data shapes:")
print("trainX.shape: {}".format(trainX.shape))
print("trainY.shape: {}".format(trainY.shape))
print("testX.shape: {}".format(testX.shape))
print("testY.shape: {}".format(testY.shape))
#%%
model = Sequential()
model.add(Conv1D(32, 1, padding = "same", input_shape = (9712, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(64, 3, padding = "same"))
model.add(Conv1D(32, 1, padding = "same"))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(128, 1, padding = "same", activation = 'relu'))
model.add(Conv1D(128, 1, padding = "same", activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(64, 3, padding = "same", activation = 'relu'))
#model.add(Conv1D(64, 3, padding = "same", activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(64, activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#%%
histo = model.fit(trainX,trainY, validation_data=(testX,testY), batch_size = 128, nb_epoch = 10)
#%%
# Final evaluation of the model
loss, accuracy = model.evaluate(testX, testY)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
