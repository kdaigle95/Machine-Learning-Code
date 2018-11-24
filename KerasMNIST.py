from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
#from quiver_engine import server
np.random.seed(1804)

#network stats
numEpochs = 20              #number of training cycles
batch_size = 64            #number of samples to be trained on
verbose = 1                 #verbosity
numOutput = 10             #number of output neurons
optimizer = Adam()           #our optimizer is Adam
numHidden = 128              #hidden layer size
validationSplit = 0.2       #percentage of training data set aside for testing
drop = 0.3

#data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
reshaped = 784

x_train = x_train.reshape(60000, reshaped).astype('float32') / 255
x_test = x_test.reshape(10000, reshaped).astype('float32') / 255


print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, numOutput)
y_test = np_utils.to_categorical(y_test, numOutput)

model = Sequential()
model.add(Dense(numHidden, input_shape=(reshaped,))) 
model.add(Activation('relu')) 
model.add(Dropout(drop))
model.add(Dense(numHidden)) 
model.add(Activation('relu'))
model.add(Dropout(drop)) 
model.add(Dense(numOutput)) 
model.add(Activation('softmax')) 
model.summary()
#server.launch(model)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

#                   training set        batch                   number of epochs   verbosity    20% of MNIST saved for testing
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = numEpochs, verbose = verbose, validation_split = validationSplit)
#server.launch(model)

score = (model.evaluate(x_test, y_test, verbose = verbose))
print("Test score:", score[0]) 
print('Test accuracy:', score[1])  
