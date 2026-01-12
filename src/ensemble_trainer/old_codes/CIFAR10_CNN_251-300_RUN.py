# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:34:58 2023

@author: melis
"""

#MODEL251

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
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
save('Model251_Train_y_labels', y_train)
save('Model251_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model251_Test_y_labels', y_test_ilk)
save('Model251_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model251_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM251.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model251_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model251_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model251_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model251_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model251_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model251_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model251_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model251 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model251 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL252

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(20 , activation="relu"))
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
save('Model252_Train_y_labels', y_train)
save('Model252_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model252_Test_y_labels', y_test_ilk)
save('Model252_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model252_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM252.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model252_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model252_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model252_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model252_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model252_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model252_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model252_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model252 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model252 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL253

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(30 , activation="relu"))
model.add(Dropout(0.5))
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
save('Model253_Train_y_labels', y_train)
save('Model253_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model253_Test_y_labels', y_test_ilk)
save('Model253_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model253_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM253.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model253_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model253_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model253_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model253_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model253_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model253_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model253_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model253 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model253 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL254

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


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
save('Model254_Train_y_labels', y_train)
save('Model254_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model254_Test_y_labels', y_test_ilk)
save('Model254_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model254_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM254.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model254_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model254_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model254_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model254_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model254_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model254_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model254_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model254 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model254 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL255

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


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
save('Model255_Train_y_labels', y_train)
save('Model255_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model255_Test_y_labels', y_test_ilk)
save('Model255_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model255_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM255.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model255_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model255_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model255_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model255_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model255_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model255_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model255_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model255 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model255 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL256

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(70 , activation="relu"))
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
save('Model256_Train_y_labels', y_train)
save('Model256_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model256_Test_y_labels', y_test_ilk)
save('Model256_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model256_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM256.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model256_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model256_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model256_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model256_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model256_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model256_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model256_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model256 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model256 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL257

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
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
save('Model257_Train_y_labels', y_train)
save('Model257_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model257_Test_y_labels', y_test_ilk)
save('Model257_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model257_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM257.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model257_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model257_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model257_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model257_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model257_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model257_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model257_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model257 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model257 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL258

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.5))
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
save('Model258_Train_y_labels', y_train)
save('Model258_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model258_Test_y_labels', y_test_ilk)
save('Model258_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model258_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM258.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model258_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model258_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model258_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model258_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model258_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model258_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model258_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model258 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model258 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL259

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(30 , activation="relu"))
#model.add(Dropout(0.5))
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
save('Model259_Train_y_labels', y_train)
save('Model259_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model259_Test_y_labels', y_test_ilk)
save('Model259_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model259_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM259.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model259_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model259_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model259_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model259_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model259_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model259_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model259_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model259 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model259 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL260

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model260_Train_y_labels', y_train)
save('Model260_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model260_Test_y_labels', y_test_ilk)
save('Model260_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model260_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM260.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model260_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model260_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model260_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model260_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model260_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model260_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model260_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model260 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model260 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL261

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.3))
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
save('Model261_Train_y_labels', y_train)
save('Model261_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model261_Test_y_labels', y_test_ilk)
save('Model261_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model261_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM261.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model261_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model261_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model261_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model261_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model261_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model261_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model261_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model261 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model261 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL262

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(50 , activation="relu"))
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
save('Model262_Train_y_labels', y_train)
save('Model262_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model262_Test_y_labels', y_test_ilk)
save('Model262_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model262_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM262.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model262_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model262_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model262_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model262_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model262_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model262_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model262_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model262 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model262 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL263

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(50 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model263_Train_y_labels', y_train)
save('Model263_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model263_Test_y_labels', y_test_ilk)
save('Model263_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model263_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM263.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model263_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model263_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model263_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model263_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model263_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model263_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model263_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model263 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model263 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL264

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
#model.add(Dropout(0.1))
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
save('Model264_Train_y_labels', y_train)
save('Model264_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model264_Test_y_labels', y_test_ilk)
save('Model264_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model264_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM264.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model264_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model264_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model264_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model264_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model264_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model264_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model264_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model264 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model264 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL265

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model265_Train_y_labels', y_train)
save('Model265_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model265_Test_y_labels', y_test_ilk)
save('Model265_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model265_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM265.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model265_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model265_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model265_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model265_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model265_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model265_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model265_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model265 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model265 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL266

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model266_Train_y_labels', y_train)
save('Model266_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model266_Test_y_labels', y_test_ilk)
save('Model266_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model266_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM266.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model266_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model266_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model266_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model266_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model266_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model266_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model266_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model266 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model266 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL267

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model267_Train_y_labels', y_train)
save('Model267_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model267_Test_y_labels', y_test_ilk)
save('Model267_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model267_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM267.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model267_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model267_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model267_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model267_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model267_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model267_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model267_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model267 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model267 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL268

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
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
save('Model268_Train_y_labels', y_train)
save('Model268_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model268_Test_y_labels', y_test_ilk)
save('Model268_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model268_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM268.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model268_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model268_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model268_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model268_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model268_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model268_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model268_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model268 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model268 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL269

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(30 , activation="relu"))
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
save('Model269_Train_y_labels', y_train)
save('Model269_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model269_Test_y_labels', y_test_ilk)
save('Model269_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model269_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM269.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model269_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model269_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model269_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model269_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model269_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model269_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model269_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model269 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model269 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL270

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
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
save('Model270_Train_y_labels', y_train)
save('Model270_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model270_Test_y_labels', y_test_ilk)
save('Model270_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model270_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM270.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model270_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model270_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model270_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model270_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model270_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model270_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model270_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model270 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model270 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL271

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(40 , activation="relu"))
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
save('Model271_Train_y_labels', y_train)
save('Model271_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model271_Test_y_labels', y_test_ilk)
save('Model271_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model271_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM271.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model271_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model271_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model271_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model271_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model271_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model271_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model271_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model271 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model271 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL272

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model272_Train_y_labels', y_train)
save('Model272_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model272_Test_y_labels', y_test_ilk)
save('Model272_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model272_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM272.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model272_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model272_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model272_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model272_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model272_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model272_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model272_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model272 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model272 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL273

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model273_Train_y_labels', y_train)
save('Model273_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model273_Test_y_labels', y_test_ilk)
save('Model273_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model273_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM273.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model273_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model273_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model273_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model273_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model273_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model273_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model273_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model273 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model273 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL274

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(60 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model274_Train_y_labels', y_train)
save('Model274_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model274_Test_y_labels', y_test_ilk)
save('Model274_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model274_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM274.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model274_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model274_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model274_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model274_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model274_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model274_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model274_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model274 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model274 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL275

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(70 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model275_Train_y_labels', y_train)
save('Model275_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model275_Test_y_labels', y_test_ilk)
save('Model275_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model275_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM275.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model275_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model275_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model275_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model275_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model275_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model275_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model275_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model275 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model275 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL276

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(70 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model276_Train_y_labels', y_train)
save('Model276_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model276_Test_y_labels', y_test_ilk)
save('Model276_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model276_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM276.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model276_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model276_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model276_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model276_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model276_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model276_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model276_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model276 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model276 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL277

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(70 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model277_Train_y_labels', y_train)
save('Model277_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model277_Test_y_labels', y_test_ilk)
save('Model277_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model277_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM277.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model277_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model277_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model277_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model277_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model277_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model277_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model277_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model277 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model277 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL278

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(80 , activation="relu"))
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
save('Model278_Train_y_labels', y_train)
save('Model278_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model278_Test_y_labels', y_test_ilk)
save('Model278_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model278_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM278.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model278_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model278_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model278_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model278_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model278_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model278_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model278_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model278 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model278 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL279

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
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
save('Model279_Train_y_labels', y_train)
save('Model279_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model279_Test_y_labels', y_test_ilk)
save('Model279_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model279_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM279.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model279_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model279_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model279_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model279_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model279_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model279_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model279_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model279 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model279 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL280

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(80 , activation="relu"))
model.add(Dropout(0.5))
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
save('Model280_Train_y_labels', y_train)
save('Model280_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model280_Test_y_labels', y_test_ilk)
save('Model280_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model280_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM280.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model280_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model280_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model280_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model280_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model280_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model280_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model280_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model280 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model280 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL281

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model281_Train_y_labels', y_train)
save('Model281_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model281_Test_y_labels', y_test_ilk)
save('Model281_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model281_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM281.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model281_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model281_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model281_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model281_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model281_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model281_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model281_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model281 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model281 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL282

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model282_Train_y_labels', y_train)
save('Model282_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model282_Test_y_labels', y_test_ilk)
save('Model282_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model282_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM282.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model282_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model282_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model282_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model282_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model282_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model282_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model282_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model282 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model282 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL283

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
model.add(Dropout(0.3))
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
save('Model283_Train_y_labels', y_train)
save('Model283_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model283_Test_y_labels', y_test_ilk)
save('Model283_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model283_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM283.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model283_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model283_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model283_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model283_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model283_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model283_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model283_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model283 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model283 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL284

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


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
save('Model284_Train_y_labels', y_train)
save('Model284_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model284_Test_y_labels', y_test_ilk)
save('Model284_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model284_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM284.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model284_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model284_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model284_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model284_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model284_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model284_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model284_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model284 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model284 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL285

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model285_Train_y_labels', y_train)
save('Model285_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model285_Test_y_labels', y_test_ilk)
save('Model285_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model285_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM285.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model285_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model285_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model285_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model285_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model285_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model285_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model285_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model285 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model285 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL286

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(50 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model286_Train_y_labels', y_train)
save('Model286_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model286_Test_y_labels', y_test_ilk)
save('Model286_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model286_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM286.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model286_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model286_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model286_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model286_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model286_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model286_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model286_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model286 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model286 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL287

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(60 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model287_Train_y_labels', y_train)
save('Model287_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model287_Test_y_labels', y_test_ilk)
save('Model287_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model287_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM287.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model287_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model287_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model287_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model287_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model287_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model287_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model287_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model287 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model287 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL288

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(60 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model288_Train_y_labels', y_train)
save('Model288_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model288_Test_y_labels', y_test_ilk)
save('Model288_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model288_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM288.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model288_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model288_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model288_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model288_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model288_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model288_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model288_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model288 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model288 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL289

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.3))
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
save('Model289_Train_y_labels', y_train)
save('Model289_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model289_Test_y_labels', y_test_ilk)
save('Model289_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model289_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM289.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model289_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model289_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model289_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model289_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model289_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model289_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model289_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model289 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model289 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL290

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model290_Train_y_labels', y_train)
save('Model290_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model290_Test_y_labels', y_test_ilk)
save('Model290_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model290_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM290.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model290_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model290_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model290_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model290_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model290_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model290_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model290_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model290 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model290 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL291

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(30 , activation="relu"))
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
save('Model291_Train_y_labels', y_train)
save('Model291_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model291_Test_y_labels', y_test_ilk)
save('Model291_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model291_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM291.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model291_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model291_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model291_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model291_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model291_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model291_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model291_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model291 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model291 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL292

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(50 , activation="relu"))
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
save('Model292_Train_y_labels', y_train)
save('Model292_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model292_Test_y_labels', y_test_ilk)
save('Model292_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model292_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM292.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model292_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model292_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model292_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model292_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model292_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model292_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model292_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model292 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model292 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL293

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(60 , activation="relu"))
model.add(Dropout(0.2))
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
save('Model293_Train_y_labels', y_train)
save('Model293_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model293_Test_y_labels', y_test_ilk)
save('Model293_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model293_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM293.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model293_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model293_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model293_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model293_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model293_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model293_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model293_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model293 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model293 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL294

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(90 , activation="relu"))
#model.add(Dropout(0.2))
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
save('Model294_Train_y_labels', y_train)
save('Model294_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model294_Test_y_labels', y_test_ilk)
save('Model294_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model294_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM294.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model294_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model294_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model294_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model294_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model294_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model294_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model294_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model294 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model294 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL295

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(100 , activation="relu"))
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
save('Model295_Train_y_labels', y_train)
save('Model295_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model295_Test_y_labels', y_test_ilk)
save('Model295_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model295_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM295.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model295_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model295_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model295_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model295_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model295_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model295_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model295_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model295 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model295 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL296

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(40 , activation="relu"))
model.add(Dropout(0.4))
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
save('Model296_Train_y_labels', y_train)
save('Model296_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model296_Test_y_labels', y_test_ilk)
save('Model296_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model296_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM296.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model296_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model296_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model296_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model296_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model296_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model296_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model296_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model296 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model296 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL297

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


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
save('Model297_Train_y_labels', y_train)
save('Model297_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model297_Test_y_labels', y_test_ilk)
save('Model297_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model297_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM297.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model297_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model297_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model297_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model297_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model297_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model297_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model297_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model297 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model297 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL298

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(70 , activation="relu"))
model.add(Dropout(0.1))
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
save('Model298_Train_y_labels', y_train)
save('Model298_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model298_Test_y_labels', y_test_ilk)
save('Model298_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model298_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM298.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model298_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model298_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model298_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model298_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model298_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model298_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model298_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model298 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model298 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL299

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


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
save('Model299_Train_y_labels', y_train)
save('Model299_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model299_Test_y_labels', y_test_ilk)
save('Model299_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model299_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM299.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model299_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model299_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model299_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model299_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model299_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model299_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model299_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model299 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model299 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL300

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
model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
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
save('Model300_Train_y_labels', y_train)
save('Model300_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model300_Test_y_labels', y_test_ilk)
save('Model300_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model300_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM300.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model300_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model300_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model300_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model300_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model300_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model300_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model300_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model300 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model300 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()