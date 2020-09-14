import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np


def xTreatment(x):
    x = np.array(x)
    x = x.astype('float32')
    x /= 255
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    return x

def yTreatment(y):
    y = np_utils.to_categorical(y, 10)
    return y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = xTreatment(x_train)
y_train = yTreatment(y_train)

x_test = xTreatment(x_test)
y_test = yTreatment(y_test)

model = Sequential()
model.add(Conv2D(filters=32, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1), kernel_size=(5,5)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, activation='relu',  kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, activation='relu',  kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))


model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=10, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rlrplateau = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3)

es = EarlyStopping(patience=10)

generator = ImageDataGenerator(
    zca_epsilon=1e-06,
    rotation_range=0.4,
    shear_range=0.3,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.33,
) # Flips, Zooms data to increase the diversity and lower overfitting

model.fit(generator.flow(x_train, y_train, batch_size=32), epochs=200, steps_per_epoch= len(x_train)//32,
          validation_steps = len(x_test), validation_data=(x_test, y_test), callbacks=[rlrplateau, es])
