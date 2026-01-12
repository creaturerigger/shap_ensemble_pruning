# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:01:34 2022

@author: melis
"""
#MODEL1

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(20 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))

test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model1_Train_y_labels', y_train)
save('Model1_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model1_Test_y_labels', y_test_ilk)
save('Model1_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model1_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM1.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model1_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model1_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model1_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model1_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model1_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model1_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model1_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model1 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model1 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#MODEL2

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(90 , activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model2_Train_y_labels', y_train)
save('Model2_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model2_Test_y_labels', y_test_ilk)
save('Model2_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model2_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM2.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model2_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model2_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model2_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model2_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model2_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model2_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model2_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model2 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model2 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL3

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(30 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model3_Train_y_labels', y_train)
save('Model3_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model3_Test_y_labels', y_test_ilk)
save('Model3_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model3_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM3.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model3_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model3_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model3_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model3_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model3_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model3_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model3_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model3 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model3 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL4

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(80 , activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model4_Train_y_labels', y_train)
save('Model4_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model4_Test_y_labels', y_test_ilk)
save('Model4_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model4_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM4.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model4_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model4_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model4_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model4_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model4_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model4_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model4_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model4 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model4 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL5

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(80 , activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model5_Train_y_labels', y_train)
save('Model5_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model5_Test_y_labels', y_test_ilk)
save('Model5_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model5_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM5.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model5_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model5_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model5_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model5_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model5_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model5_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model5_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model5 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model5 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#MODEL6

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))


test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model6_Train_y_labels', y_train)
save('Model6_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model6_Test_y_labels', y_test_ilk)
save('Model6_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model6_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM6.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model6_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model6_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model6_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model6_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model6_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model6_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model6_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model6 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model6 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#MODEL7

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(20 , activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))


test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model7_Train_y_labels', y_train)
save('Model7_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model7_Test_y_labels', y_test_ilk)
save('Model7_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model7_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM7.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model7_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model7_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model7_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model7_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model7_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model7_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model7_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model7 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model7 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL8

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score


from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(30 , activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))


test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()


cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model8_Train_y_labels', y_train)
save('Model8_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model8_Test_y_labels', y_test_ilk)
save('Model8_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model8_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM8.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model8_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model8_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model8_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model8_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model8_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model8_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model8_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model8 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model8 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#MODEL9
import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(30 , activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))


test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model9_Train_y_labels', y_train)
save('Model9_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model9_Test_y_labels', y_test_ilk)
save('Model9_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model9_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM9.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model9_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model9_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model9_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model9_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model9_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model9_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model9_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model9 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model9 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL10

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))


test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model10_Train_y_labels', y_train)
save('Model10_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model10_Test_y_labels', y_test_ilk)
save('Model10_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model10_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM10.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model10_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model10_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model10_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model10_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model10_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model10_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model10_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model10 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model10 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL11

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(50 , activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model11_Train_y_labels', y_train)
save('Model11_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model11_Test_y_labels', y_test_ilk)
save('Model11_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model11_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM11.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model11_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model11_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model11_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model11_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model11_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model11_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model11_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model11 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model11 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL12

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(60 , activation="relu"))
#model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model12_Train_y_labels', y_train)
save('Model12_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model12_Test_y_labels', y_test_ilk)
save('Model12_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model12_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM12.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model12_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model12_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model12_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model12_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model12_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model12_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model12_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model12 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model12 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL13

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(60 , activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model13_Train_y_labels', y_train)
save('Model13_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model13_Test_y_labels', y_test_ilk)
save('Model13_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model13_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM13.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model13_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model13_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model13_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model13_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model13_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model13_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model13_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model13 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model13 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL14

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(80 , activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model14_Train_y_labels', y_train)
save('Model14_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model14_Test_y_labels', y_test_ilk)
save('Model14_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model14_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM14.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model14_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model14_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model14_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model14_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model14_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model14_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model14_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model14 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model14 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL15

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(90 , activation="relu"))
#model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model15_Train_y_labels', y_train)
save('Model15_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model15_Test_y_labels', y_test_ilk)
save('Model15_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model15_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM15.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model15_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model15_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model15_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model15_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model15_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model15_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model15_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model15 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model15 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL16

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model16_Train_y_labels', y_train)
save('Model16_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model16_Test_y_labels', y_test_ilk)
save('Model16_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model16_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM16.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model16_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model16_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model16_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model16_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model16_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model16_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model16_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model16 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model16 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL17

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model17_Train_y_labels', y_train)
save('Model17_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model17_Test_y_labels', y_test_ilk)
save('Model17_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model17_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM17.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model17_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model17_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model17_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model17_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model17_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model17_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model17_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model17 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model17 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL18

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(70 , activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model18_Train_y_labels', y_train)
save('Model18_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model18_Test_y_labels', y_test_ilk)
save('Model18_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model18_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM18.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model18_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model18_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model18_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model18_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model18_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model18_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model18_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model18 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model18 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL19

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model19_Train_y_labels', y_train)
save('Model19_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model19_Test_y_labels', y_test_ilk)
save('Model19_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model19_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM19.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model19_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model19_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model19_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model19_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model19_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model19_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model19_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model19 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model19 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL20

import pandas as pd
import keras
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import cifar10
 # load dataset
(x_train, y_train), (x_test, y_test_ilk) = cifar10.load_data()

# reshape dataset to have a single channel
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

x_train = x_train / 255
x_test = x_test /255

y_train_categoric = to_categorical(y_train)
y_test = to_categorical(y_test_ilk)

  
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(20 , activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(10, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])


history = model.fit(x_train, y_train_categoric, epochs=50, validation_data=(x_test,y_test))



test_predictions_probs = model.predict(x_test)
test_predictions_labels = pd.DataFrame(model.predict(x_test))
test_predictions_labels = pd.DataFrame(test_predictions_labels.idxmax(axis= 1))

train_predictions_probs = pd.DataFrame(model.predict(x_train))
train_predictions_probs = train_predictions_probs.to_numpy()

"""
print(np.argmax(np.round(predictions[0])))
"""

cm = confusion_matrix(y_test_ilk, test_predictions_labels)
accuracy_score = accuracy_score(y_test_ilk, test_predictions_labels)
mae = mean_absolute_error(y_test_ilk, test_predictions_labels)
mape = mean_absolute_percentage_error(y_test_ilk, test_predictions_labels)
mse = mean_squared_error(y_test_ilk, test_predictions_labels)
Performance_Scores = (accuracy_score, mae , mape, mse )

from numpy import save
save('Model20_Train_y_labels', y_train)
save('Model20_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model20_Test_y_labels', y_test_ilk)
save('Model20_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model20_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM20.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model20_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model20_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model20_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model20_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model20_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model20_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model20_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model20 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model20 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()