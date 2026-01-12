# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:55:14 2023

@author: melis
"""

#MODEL31

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(20 , activation="relu"))
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
save('Model31_Train_y_labels', y_train)
save('Model31_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model31_Test_y_labels', y_test_ilk)
save('Model31_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model31_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM31.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model31_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model31_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model31_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model31_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model31_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model31_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model31_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model31 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model31 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL32

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(40 , activation="relu"))
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
save('Model32_Train_y_labels', y_train)
save('Model32_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model32_Test_y_labels', y_test_ilk)
save('Model32_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model32_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM32.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model32_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model32_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model32_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model32_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model32_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model32_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model32_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model32 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model32 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL33

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(50 , activation="relu"))
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
save('Model33_Train_y_labels', y_train)
save('Model33_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model33_Test_y_labels', y_test_ilk)
save('Model33_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model33_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM33.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model33_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model33_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model33_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model33_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model33_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model33_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model33_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model33 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model33 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL34

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(50 , activation="relu"))
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
save('Model34_Train_y_labels', y_train)
save('Model34_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model34_Test_y_labels', y_test_ilk)
save('Model34_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model34_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM34.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model34_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model34_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model34_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model34_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model34_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model34_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model34_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model34 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model34 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL35

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
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
save('Model35_Train_y_labels', y_train)
save('Model35_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model35_Test_y_labels', y_test_ilk)
save('Model35_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model35_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM35.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model35_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model35_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model35_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model35_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model35_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model35_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model35_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model35 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model35 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL36

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

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
save('Model36_Train_y_labels', y_train)
save('Model36_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model36_Test_y_labels', y_test_ilk)
save('Model36_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model36_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM36.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model36_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model36_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model36_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model36_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model36_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model36_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model36_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model36 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model36 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL37

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
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
save('Model37_Train_y_labels', y_train)
save('Model37_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model37_Test_y_labels', y_test_ilk)
save('Model37_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model37_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM37.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model37_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model37_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model37_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model37_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model37_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model37_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model37_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model37 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model37 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL38

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
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
save('Model38_Train_y_labels', y_train)
save('Model38_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model38_Test_y_labels', y_test_ilk)
save('Model38_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model38_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM38.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model38_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model38_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model38_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model38_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model38_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model38_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model38_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model38 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model38 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL39

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

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(96, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100 , activation="relu"))
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
save('Model39_Train_y_labels', y_train)
save('Model39_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model39_Test_y_labels', y_test_ilk)
save('Model39_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model39_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM39.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model39_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model39_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model39_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model39_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model39_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model39_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model39_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model39 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model39 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL40

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

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
save('Model40_Train_y_labels', y_train)
save('Model40_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model40_Test_y_labels', y_test_ilk)
save('Model40_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model40_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM40.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model40_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model40_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model40_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model40_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model40_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model40_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model40_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model40 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model40 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL41

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(40 , activation="relu"))
#model.add(Dropout(0.3))
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
save('Model41_Train_y_labels', y_train)
save('Model41_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model41_Test_y_labels', y_test_ilk)
save('Model41_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model41_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM41.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model41_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model41_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model41_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model41_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model41_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model41_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model41_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model41 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model41 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL42

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

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
save('Model42_Train_y_labels', y_train)
save('Model42_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model42_Test_y_labels', y_test_ilk)
save('Model42_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model42_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM42.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model42_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model42_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model42_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model42_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model42_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model42_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model42_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model42 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model42 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL43

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(70 , activation="relu"))
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
save('Model43_Train_y_labels', y_train)
save('Model43_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model43_Test_y_labels', y_test_ilk)
save('Model43_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model43_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM43.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model43_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model43_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model43_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model43_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model43_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model43_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model43_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model43 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model43 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL44

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

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
save('Model44_Train_y_labels', y_train)
save('Model44_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model44_Test_y_labels', y_test_ilk)
save('Model44_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model44_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM44.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model44_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model44_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model44_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model44_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model44_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model44_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model44_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model44 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model44 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL45

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100 , activation="relu"))
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
save('Model45_Train_y_labels', y_train)
save('Model45_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model45_Test_y_labels', y_test_ilk)
save('Model45_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model45_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM45.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model45_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model45_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model45_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model45_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model45_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model45_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model45_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model45 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model45 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL46

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(20 , activation="relu"))
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
save('Model46_Train_y_labels', y_train)
save('Model46_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model46_Test_y_labels', y_test_ilk)
save('Model46_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model46_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM46.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model46_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model46_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model46_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model46_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model46_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model46_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model46_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model46 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model46 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL47

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

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
save('Model47_Train_y_labels', y_train)
save('Model47_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model47_Test_y_labels', y_test_ilk)
save('Model47_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model47_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM47.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model47_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model47_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model47_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model47_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model47_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model47_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model47_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model47 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model47 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL48

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(50 , activation="relu"))
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
save('Model48_Train_y_labels', y_train)
save('Model48_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model48_Test_y_labels', y_test_ilk)
save('Model48_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model48_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM48.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model48_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model48_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model48_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model48_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model48_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model48_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model48_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model48 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model48 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL49

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
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
save('Model49_Train_y_labels', y_train)
save('Model49_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model49_Test_y_labels', y_test_ilk)
save('Model49_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model49_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM49.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model49_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model49_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model49_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model49_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model49_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model49_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model49_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model49 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model49 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL50

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

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(90 , activation="relu"))
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
save('Model50_Train_y_labels', y_train)
save('Model50_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model50_Test_y_labels', y_test_ilk)
save('Model50_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model50_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM50.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model50_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model50_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model50_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model50_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model50_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model50_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model50_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model50 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model50 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

