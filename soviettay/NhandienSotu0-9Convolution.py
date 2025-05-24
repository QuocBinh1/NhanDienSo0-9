#nhận dạng số viết tay từ 0 --> 9
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D , Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K


model = tf.keras.Sequential([
    Conv2D(filters=16 , kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Flatten(),

    Dense(300, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255


y_train =to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# print(y_train[2], 'train samples')


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1) 
img_test = cv2.imread('/anh8.png',0)

img_test = img_test.reshape(1, 28, 28, 1)
print(model.predict(img_test))


