import SetEnvForGpu

import itertools

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as prc
import sklearn.cross_validation as scv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D

from keras.utils import np_utils

import DataTools
import ImageUtils
import Models

def showImg(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Network params
nb_classes = 11
nb_epoch = 12

img_dim_x=100
img_dim_y=100
input_shape = (3,img_dim_y,img_dim_x)

model = Models.getUNet(input_shape, nb_classes)

testId = '6100_1_3'
(img_i, mask) = DataTools.loadAll(testId)
img = img_i.astype(float)
img = prc.scale(img.reshape(-1, 1)).reshape(img_i.shape)

gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 10)
gg = itertools.islice(gall, 2000)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gg, img, mask)

(x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, masks, test_size=0.2)
y_train_cat = np_utils.to_categorical(y_train.flatten(), nb_classes)
y_train_cat = y_train_cat.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2], nb_classes))

model.fit(x_train, y_train, nb_epoch=10)

y_cv_cat = np_utils.to_categorical(y_cv.flatten(), nb_classes)
y_cv_cat = y_cv_cat.reshape((y_cv.shape[0], y_cv.shape[1]*y_cv.shape[2], nb_classes))

rez = model.predict(x_cv)
xx = np.argmax(rez, axis = 2)
xx = xx.reshape((-1, img_dim_y, img_dim_x))
