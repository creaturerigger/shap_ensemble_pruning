# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:54:11 2023

@author: melis
"""

#MODEL21

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
save('Model21_Train_y_labels', y_train)
save('Model21_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model21_Test_y_labels', y_test_ilk)
save('Model21_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model21_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM21.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model21_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model21_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model21_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model21_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model21_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model21_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model21_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model21 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model21 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL22

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
save('Model22_Train_y_labels', y_train)
save('Model22_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model22_Test_y_labels', y_test_ilk)
save('Model22_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model22_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM22.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model22_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model22_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model22_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model22_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model22_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model22_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model22_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model22 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model22 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL23

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
save('Model23_Train_y_labels', y_train)
save('Model23_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model23_Test_y_labels', y_test_ilk)
save('Model23_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model23_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM23.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model23_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model23_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model23_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model23_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model23_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model23_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model23_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model23 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model23 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL24

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
save('Model24_Train_y_labels', y_train)
save('Model24_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model24_Test_y_labels', y_test_ilk)
save('Model24_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model24_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM24.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model24_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model24_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model24_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model24_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model24_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model24_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model24_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model24 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model24 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL25

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
save('Model25_Train_y_labels', y_train)
save('Model25_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model25_Test_y_labels', y_test_ilk)
save('Model25_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model25_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM25.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model25_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model25_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model25_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model25_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model25_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model25_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model25_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model25 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model25 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL26

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
save('Model26_Train_y_labels', y_train)
save('Model26_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model26_Test_y_labels', y_test_ilk)
save('Model26_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model26_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM26.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model26_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model26_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model26_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model26_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model26_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model26_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model26_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model26 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model26 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL27

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
save('Model27_Train_y_labels', y_train)
save('Model27_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model27_Test_y_labels', y_test_ilk)
save('Model27_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model27_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM27.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model27_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model27_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model27_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model27_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model27_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model27_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model27_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model27 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model27 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL28

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
save('Model28_Train_y_labels', y_train)
save('Model28_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model28_Test_y_labels', y_test_ilk)
save('Model28_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model28_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM28.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model28_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model28_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model28_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model28_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model28_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model28_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model28_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model28 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model28 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL29

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

model.add(Dense(80 , activation="relu"))
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
save('Model29_Train_y_labels', y_train)
save('Model29_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model29_Test_y_labels', y_test_ilk)
save('Model29_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model29_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM29.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model29_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model29_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model29_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model29_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model29_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model29_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model29_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model29 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model29 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#MODEL30

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

model.add(Dense(90 , activation="relu"))
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
save('Model30_Train_y_labels', y_train)
save('Model30_Train_Predictions_withProbabilities', train_predictions_probs)
save('Model30_Test_y_labels', y_test_ilk)
save('Model30_Test_Predictions_withProbabilties', test_predictions_probs)
save('Model30_Test_Predictions_final', test_predictions_labels)

pd.DataFrame(cm).to_csv('CM30.csv') #,index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_train).to_csv('Model30_Train_y_labels.csv') #,index_label = 'Index', header = ['Train_Labels'])
pd.DataFrame(train_predictions_probs).to_csv('Model30_Train_Predictions_withProbabilities.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(y_test_ilk).to_csv('Model30_Test_y_labels.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(test_predictions_probs).to_csv('Model30_Test_Predictions_withProbabilties.csv') #, index_label = 'Index', header = ['0','1','2','3','4','5','6','7','8','9'])
pd.DataFrame(test_predictions_labels).to_csv('Model30_Test_Predictions_final.csv') #,index_label = 'Index', header = ['Test_Labels'])
pd.DataFrame(Performance_Scores).to_csv('Model30_Performance_Scores.csv') #, index_label = 'Index', header = ['Performans_Scores'])

def myprint(s):
    with open('Model30_Summary.txt','a') as f:
        print(s, file=f)
model.summary(print_fn=myprint)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model30 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model30 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()