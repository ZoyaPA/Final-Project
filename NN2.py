from tensorflow.keras.datasets import cifar10


from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
#from keras.models import Sequential, Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM

from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

data = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train[1].shape
X_train.shape
X_test.shape

import matplotlib.pyplot as plt

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

f, axarr = plt.subplots(1, 5)
f.set_size_inches(4, 1.5)
plt.show()

images = X_train
images.shape


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).reshape(32,32,1)


img = images[49999]
gray = rgb2gray(img)
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()

gray_images = np.zeros((50000, 32, 32,1))

for i in range(0, 500):
    gray_images[i] = rgb2gray(images[i])
gray_images[0].shape


plt.imshow(images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

gray_images[49999] = rgb2gray(images[49999])

plt.imshow(gray_images[49999], cmap='gray', vmin=0, vmax=255)
plt.show()

type(gray_images)


#np.save("gray_images",gray_images)
gray_images=np.load('gray_images.npy', mmap_mode='r')
aaa.shape


plt.imshow(aaa[49998], cmap='gray', vmin=0, vmax=255)
plt.show()

images.shape
pixels = images.flatten().reshape(50000,3072 )
print
pixels.shape

model = Sequential()
model.add(InputLayer(input_shape=(32, 32, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
#model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
model.add(Flatten())
model.add(Dense(10000))
model.add(Dense(3072))

# Finish model
model.compile(optimizer='adam',loss='mse')

#Train the neural network
model.fit(x=gray_images[0:500]/255, y=pixels[0:500]/255, batch_size=500, epochs=1000)

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=gray_images[0:500]/255, y=pixels[0:500]/255, batch_size=500, epochs=1000)

model.compile(optimizer='SGD',loss='mse')
model.fit(x=gray_images[0:500]/255, y=pixels[0:500]/255, batch_size=500, epochs=1000)

model.compile(optimizer='ftrl',loss='mse')
model.fit(x=gray_images[0:500]/255, y=pixels[0:500]/255, batch_size=500, epochs=1000)

model.compile(optimizer='adagrad',loss='mse')
model.fit(x=gray_images[0:500]/255, y=pixels[0:500]/255, batch_size=500, epochs=1000)


#print(model.evaluate(X, Y, batch_size=1))
# Output colorizations
output = model.predict(gray_images[501:502]/255)


plt.imshow(images[501], vmin=0, vmax=1)
#plt.imshow(output[0].reshape(32,32,3), vmin=0, vmax=1)
plt.show()
print(output[0]*255)#-images[501])

output.shape

2