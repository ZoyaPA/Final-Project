import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l1
from tensorflow.keras.datasets import cifar10
from keras.layers import Conv2D, Flatten,concatenate
import matplotlib.pyplot as plt

gray_images=np.load('gray_images1.npy', mmap_mode='r')
gray_images.shape
data = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
images = X_train
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
for i in range(0,50000,500):
    print(i)
    model.fit(x=[y_train_10[i:i+500],gray_images[i:i+500]/255],y=pixels[i:i+500]/255, batch_size=500, epochs=10)
pixels.shape
check = model.predict([y_train_10[700:701],gray_images[700:701]/255])
check[0]



plt.imshow(images[700], vmin=0, vmax=1)
plt.show()

plt.imshow(check[0].reshape(32,32,3), vmin=0, vmax=1)
plt.show()
