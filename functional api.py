import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l1
from tensorflow.keras.datasets import cifar10
from keras.layers import Conv2D, Flatten,concatenate

gray_images=np.load('gray_images1.npy', mmap_mode='r')
gray_images.shape
data = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train[1].shape
X_train.shape
X_test.shape

pixels = X_train.flatten().reshape(50000,3072 )
y_train_10= np.zeros((50000,10))
y_train_10[np.arange(y_train.size),y_train.reshape(50000)]=1
y_train_10[1]
y_train_10.shape

inputA = Input(shape=(10,))
inputB = Input(shape=(32,32,1,))

conv_1 = Conv2D(32, (3, 3), activation='relu')(inputB)
conv_2 = Conv2D(64, (3, 3), activation='relu')(conv_1)
conv_3 =Conv2D(64, (3, 3), activation='relu')(conv_2)
flatten = Flatten()(conv_3)
combined = concatenate([flatten, inputA])

output = Dense(3072,activation='relu')(combined)

model = Model(inputs=[inputA,inputB], outputs=output)
model.summary()
model.compile(optimizer='adam',loss='mse')

y_train_10[0:500].shape
model.fit(x=[y_train_10[0:500],gray_images[0:500]/255],y=pixels[0:500]/255, batch_size=500, epochs=2000)
pixels.shape
check = model.predict([y_train_10[502:503],gray_images[502:503]/255])
plt.imshow(images[502], vmin=0, vmax=1)
#plt.imshow(check[0].reshape(32,32,3), vmin=0, vmax=1)
plt.show()