
import SetEnvForGpu

import itertools

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as prc
import sklearn.model_selection as scv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D

from keras.utils import np_utils

import DataTools
import ImageUtils

def showImg(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Network params
batch_size = 20
nb_classes = 10
nb_epoch = 12

img_dim_x=51
img_dim_y=51
input_shape = (3,img_dim_y,img_dim_x)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


testId = '6100_1_3'
(img_i, mask) = DataTools.loadAll(testId)
img = img_i.astype(float)
img = prc.scale(img.reshape(-1, 1)).reshape(img_i.shape)

gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 10)
gg = itertools.islice(gall, 2000)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)

(x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, classes, test_size=0.2)
y_train_cat = np_utils.to_categorical(y_train-1, nb_classes)
y_cv_cat = np_utils.to_categorical(y_cv-1, nb_classes)

model.fit(x_train, y_train_cat, nb_epoch=30)
#y_pred = model.predict(x_cv)

score, acc = model.evaluate(x_cv, y_cv_cat)
print("Accuracy on CV dataset: {0}".format(acc))
