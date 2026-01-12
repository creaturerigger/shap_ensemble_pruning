# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:37:10 2023

@author: melis
"""

#MODEL201

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
save('Model201_Train_y_labels', y_train)
save('Model201_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model201_Test_y_labels', y_test_ilk)
save('Model201_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model201_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM201.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model201_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model201_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model201_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model201_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model201_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model201_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model201_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model201 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model201 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL202

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
save('Model202_Train_y_labels', y_train)
save('Model202_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model202_Test_y_labels', y_test_ilk)
save('Model202_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model202_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM202.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model202_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model202_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model202_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model202_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model202_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model202_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model202_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model202 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model202 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL203

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
save('Model203_Train_y_labels', y_train)
save('Model203_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model203_Test_y_labels', y_test_ilk)
save('Model203_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model203_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM203.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model203_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model203_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model203_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model203_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model203_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model203_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model203_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model203 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model203 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL204

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
save('Model204_Train_y_labels', y_train)
save('Model204_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model204_Test_y_labels', y_test_ilk)
save('Model204_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model204_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM204.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model204_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model204_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model204_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model204_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model204_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model204_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model204_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model204 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model204 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL205

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
save('Model205_Train_y_labels', y_train)
save('Model205_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model205_Test_y_labels', y_test_ilk)
save('Model205_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model205_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM205.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model205_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model205_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model205_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model205_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model205_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model205_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model205_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model205 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model205 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL206

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
save('Model206_Train_y_labels', y_train)
save('Model206_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model206_Test_y_labels', y_test_ilk)
save('Model206_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model206_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM206.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model206_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model206_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model206_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model206_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model206_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model206_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model206_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model206 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model206 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL207

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
save('Model207_Train_y_labels', y_train)
save('Model207_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model207_Test_y_labels', y_test_ilk)
save('Model207_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model207_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM207.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model207_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model207_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model207_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model207_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model207_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model207_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model207_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model207 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model207 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL208

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
save('Model208_Train_y_labels', y_train)
save('Model208_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model208_Test_y_labels', y_test_ilk)
save('Model208_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model208_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM208.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model208_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model208_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model208_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model208_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model208_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model208_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model208_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model208 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model208 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL209

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
save('Model209_Train_y_labels', y_train)
save('Model209_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model209_Test_y_labels', y_test_ilk)
save('Model209_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model209_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM209.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model209_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model209_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model209_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model209_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model209_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model209_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model209_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model209 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model209 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL210

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
save('Model210_Train_y_labels', y_train)
save('Model210_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model210_Test_y_labels', y_test_ilk)
save('Model210_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model210_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM210.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model210_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model210_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model210_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model210_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model210_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model210_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model210_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model210 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model210 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL211

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
save('Model211_Train_y_labels', y_train)
save('Model211_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model211_Test_y_labels', y_test_ilk)
save('Model211_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model211_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM211.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model211_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model211_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model211_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model211_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model211_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model211_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model211_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model211 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model211 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL212

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
model.add(Dropout(0.3))
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
save('Model212_Train_y_labels', y_train)
save('Model212_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model212_Test_y_labels', y_test_ilk)
save('Model212_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model212_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM212.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model212_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model212_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model212_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model212_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model212_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model212_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model212_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model212 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model212 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL213

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
save('Model213_Train_y_labels', y_train)
save('Model213_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model213_Test_y_labels', y_test_ilk)
save('Model213_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model213_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM213.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model213_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model213_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model213_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model213_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model213_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model213_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model213_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model213 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model213 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL214

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
save('Model214_Train_y_labels', y_train)
save('Model214_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model214_Test_y_labels', y_test_ilk)
save('Model214_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model214_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM214.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model214_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model214_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model214_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model214_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model214_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model214_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model214_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model214 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model214 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL215

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
save('Model215_Train_y_labels', y_train)
save('Model215_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model215_Test_y_labels', y_test_ilk)
save('Model215_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model215_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM215.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model215_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model215_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model215_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model215_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model215_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model215_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model215_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model215 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model215 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL216

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
save('Model216_Train_y_labels', y_train)
save('Model216_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model216_Test_y_labels', y_test_ilk)
save('Model216_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model216_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM216.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model216_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model216_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model216_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model216_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model216_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model216_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model216_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model216 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model216 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL217

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
#model.add(Dropout(0.1))
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
save('Model217_Train_y_labels', y_train)
save('Model217_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model217_Test_y_labels', y_test_ilk)
save('Model217_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model217_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM217.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model217_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model217_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model217_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model217_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model217_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model217_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model217_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model217 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model217 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL218

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
save('Model218_Train_y_labels', y_train)
save('Model218_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model218_Test_y_labels', y_test_ilk)
save('Model218_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model218_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM218.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model218_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model218_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model218_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model218_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model218_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model218_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model218_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model218 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model218 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL219

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
save('Model219_Train_y_labels', y_train)
save('Model219_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model219_Test_y_labels', y_test_ilk)
save('Model219_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model219_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM219.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model219_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model219_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model219_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model219_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model219_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model219_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model219_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model219 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model219 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL220

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
save('Model220_Train_y_labels', y_train)
save('Model220_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model220_Test_y_labels', y_test_ilk)
save('Model220_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model220_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM220.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model220_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model220_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model220_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model220_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model220_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model220_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model220_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model220 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model220 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL221

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
#model.add(Dropout(0.2))
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
save('Model221_Train_y_labels', y_train)
save('Model221_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model221_Test_y_labels', y_test_ilk)
save('Model221_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model221_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM221.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model221_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model221_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model221_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model221_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model221_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model221_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model221_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model221 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model221 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL222

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
#model.add(Dropout(0.2))
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
save('Model222_Train_y_labels', y_train)
save('Model222_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model222_Test_y_labels', y_test_ilk)
save('Model222_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model222_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM222.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model222_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model222_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model222_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model222_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model222_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model222_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model222_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model222 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model222 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL223

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
save('Model223_Train_y_labels', y_train)
save('Model223_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model223_Test_y_labels', y_test_ilk)
save('Model223_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model223_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM223.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model223_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model223_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model223_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model223_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model223_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model223_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model223_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model223 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model223 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL224

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
save('Model224_Train_y_labels', y_train)
save('Model224_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model224_Test_y_labels', y_test_ilk)
save('Model224_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model224_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM224.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model224_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model224_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model224_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model224_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model224_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model224_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model224_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model224 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model224 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL225

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
save('Model225_Train_y_labels', y_train)
save('Model225_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model225_Test_y_labels', y_test_ilk)
save('Model225_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model225_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM225.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model225_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model225_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model225_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model225_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model225_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model225_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model225_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model225 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model225 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL226

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
save('Model226_Train_y_labels', y_train)
save('Model226_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model226_Test_y_labels', y_test_ilk)
save('Model226_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model226_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM226.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model226_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model226_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model226_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model226_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model226_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model226_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model226_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model226 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model226 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL227

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
save('Model227_Train_y_labels', y_train)
save('Model227_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model227_Test_y_labels', y_test_ilk)
save('Model227_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model227_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM227.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model227_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model227_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model227_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model227_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model227_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model227_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model227_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model227 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model227 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL228

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
save('Model228_Train_y_labels', y_train)
save('Model228_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model228_Test_y_labels', y_test_ilk)
save('Model228_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model228_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM228.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model228_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model228_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model228_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model228_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model228_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model228_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model228_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model228 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model228 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL229

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
save('Model229_Train_y_labels', y_train)
save('Model229_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model229_Test_y_labels', y_test_ilk)
save('Model229_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model229_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM229.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model229_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model229_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model229_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model229_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model229_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model229_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model229_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model229 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model229 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL230

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
save('Model230_Train_y_labels', y_train)
save('Model230_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model230_Test_y_labels', y_test_ilk)
save('Model230_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model230_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM230.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model230_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model230_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model230_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model230_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model230_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model230_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model230_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model230 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model230 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL231

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
save('Model231_Train_y_labels', y_train)
save('Model231_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model231_Test_y_labels', y_test_ilk)
save('Model231_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model231_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM231.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model231_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model231_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model231_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model231_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model231_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model231_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model231_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model231 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model231 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL232

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
save('Model232_Train_y_labels', y_train)
save('Model232_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model232_Test_y_labels', y_test_ilk)
save('Model232_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model232_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM232.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model232_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model232_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model232_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model232_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model232_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model232_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model232_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model232 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model232 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL233

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
save('Model233_Train_y_labels', y_train)
save('Model233_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model233_Test_y_labels', y_test_ilk)
save('Model233_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model233_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM233.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model233_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model233_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model233_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model233_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model233_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model233_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model233_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model233 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model233 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL234

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
save('Model234_Train_y_labels', y_train)
save('Model234_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model234_Test_y_labels', y_test_ilk)
save('Model234_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model234_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM234.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model234_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model234_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model234_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model234_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model234_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model234_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model234_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model234 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model234 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL235

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
#model.add(Dropout(0.4))
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
save('Model235_Train_y_labels', y_train)
save('Model235_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model235_Test_y_labels', y_test_ilk)
save('Model235_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model235_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM235.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model235_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model235_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model235_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model235_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model235_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model235_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model235_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model235 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model235 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL236

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
save('Model236_Train_y_labels', y_train)
save('Model236_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model236_Test_y_labels', y_test_ilk)
save('Model236_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model236_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM236.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model236_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model236_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model236_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model236_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model236_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model236_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model236_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model236 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model236 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL237

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
save('Model237_Train_y_labels', y_train)
save('Model237_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model237_Test_y_labels', y_test_ilk)
save('Model237_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model237_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM237.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model237_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model237_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model237_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model237_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model237_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model237_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model237_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model237 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model237 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL238

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
save('Model238_Train_y_labels', y_train)
save('Model238_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model238_Test_y_labels', y_test_ilk)
save('Model238_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model238_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM238.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model238_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model238_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model238_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model238_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model238_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model238_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model238_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model238 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model238 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL239

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
save('Model239_Train_y_labels', y_train)
save('Model239_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model239_Test_y_labels', y_test_ilk)
save('Model239_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model239_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM239.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model239_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model239_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model239_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model239_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model239_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model239_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model239_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model239 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model239 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL240

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
save('Model240_Train_y_labels', y_train)
save('Model240_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model240_Test_y_labels', y_test_ilk)
save('Model240_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model240_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM240.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model240_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model240_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model240_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model240_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model240_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model240_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model240_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model240 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model240 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL241

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
save('Model241_Train_y_labels', y_train)
save('Model241_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model241_Test_y_labels', y_test_ilk)
save('Model241_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model241_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM241.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model241_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model241_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model241_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model241_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model241_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model241_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model241_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model241 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model241 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL242

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
save('Model242_Train_y_labels', y_train)
save('Model242_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model242_Test_y_labels', y_test_ilk)
save('Model242_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model242_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM242.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model242_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model242_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model242_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model242_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model242_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model242_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model242_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model242 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model242 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL243

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
save('Model243_Train_y_labels', y_train)
save('Model243_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model243_Test_y_labels', y_test_ilk)
save('Model243_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model243_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM243.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model243_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model243_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model243_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model243_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model243_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model243_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model243_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model243 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model243 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL244

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

model.add(Dense(20 , activation="relu"))
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
save('Model244_Train_y_labels', y_train)
save('Model244_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model244_Test_y_labels', y_test_ilk)
save('Model244_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model244_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM244.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model244_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model244_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model244_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model244_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model244_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model244_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model244_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model244 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model244 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL245

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
save('Model245_Train_y_labels', y_train)
save('Model245_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model245_Test_y_labels', y_test_ilk)
save('Model245_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model245_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM245.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model245_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model245_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model245_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model245_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model245_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model245_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model245_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model245 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model245 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL246

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
save('Model246_Train_y_labels', y_train)
save('Model246_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model246_Test_y_labels', y_test_ilk)
save('Model246_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model246_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM246.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model246_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model246_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model246_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model246_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model246_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model246_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model246_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model246 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model246 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL247

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
save('Model247_Train_y_labels', y_train)
save('Model247_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model247_Test_y_labels', y_test_ilk)
save('Model247_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model247_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM247.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model247_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model247_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model247_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model247_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model247_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model247_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model247_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model247 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model247 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL248

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
save('Model248_Train_y_labels', y_train)
save('Model248_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model248_Test_y_labels', y_test_ilk)
save('Model248_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model248_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM248.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model248_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model248_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model248_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model248_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model248_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model248_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model248_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model248 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model248 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL249

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
save('Model249_Train_y_labels', y_train)
save('Model249_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model249_Test_y_labels', y_test_ilk)
save('Model249_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model249_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM249.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model249_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model249_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model249_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model249_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model249_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model249_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model249_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model249 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model249 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL250

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
model.add(Dropout(0.3))
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
save('Model250_Train_y_labels', y_train)
save('Model250_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model250_Test_y_labels', y_test_ilk)
save('Model250_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model250_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM250.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model250_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model250_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model250_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model250_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model250_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model250_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model250_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model250 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model250 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()