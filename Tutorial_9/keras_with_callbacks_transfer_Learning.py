import keras
from keras.datasets import mnist
from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 120
num_classess = 10
epochs = 1

# input dimensions
img_rows, img_cols = 28,28

# dataset input
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data initialization
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classess)
y_test = keras.utils.to_categorical(y_test, num_classess)

# Model (forward)
my_model = models.Sequential()
my_model.add(Conv2D(32, kernel_size=(3,3),activation='relu', input_shape = input_shape))
my_model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Flatten())
my_model.add(Dense(256, activation='relu'))
my_model.add(Dropout(0.02))
my_model.add(Dense(num_classess,activation='softmax'))

# Transfer Learning

my_model.load_weights('weights.hdf5')

# Model (backward)
my_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])



# Feed Data
my_model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs, validation_data=(x_test,y_test))

