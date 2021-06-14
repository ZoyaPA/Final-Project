from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np

data=cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train[1].shape
X_train.shape

import matplotlib.pyplot as plt
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

f, axarr = plt.subplots(1, 5)
f.set_size_inches(4, 1.5)
plt.show()

for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)

images= X_train
images.shape

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = images[12]
gray = rgb2gray(img)
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()

gray_images=np.zeros((50000,32,32))


for i in range(0,400):
    gray_images[i]=rgb2gray(images[i])

gray_images[1].shape

plt.imshow(gray_images[399], cmap='gray', vmin=0, vmax=255)
plt.show()
images[399]