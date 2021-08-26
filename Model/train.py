# import all the libraries
import tensorflow
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist

# loading the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

classes = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)

#normalize the image

X_train, X_test = X_train/255, X_test/255

# 1. Model architecture

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128,  kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform')) #hidden 1
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid')) #hidden 2
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform')) #hidden 2
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax')) #output layer

model.compile(optimizer="Adam", loss=tensorflow.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32)

model.evaluate(X_test, y_test)