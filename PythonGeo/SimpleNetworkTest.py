
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
import Models

def showImg(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Network params
nb_classes = 11
nb_epoch = 12

img_dim_x=51
img_dim_y=51
input_shape = (3,img_dim_y,img_dim_x)

model = Models.getSimpleModel(nb_classes, input_shape)

testId = '6100_1_3'
(img_i, mask) = DataTools.loadAll(testId)
img = img_i.astype(float)
img = prc.scale(img.reshape(-1, 1)).reshape(img_i.shape)

gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 10)
gg = itertools.islice(gall, 2000)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)

(x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, classes, test_size=0.2)
y_train_cat = np_utils.to_categorical(y_train, nb_classes)
y_cv_cat = np_utils.to_categorical(y_cv, nb_classes)

model.fit(x_train, y_train_cat, nb_epoch=15)
#y_pred = model.predict(x_cv)

score, acc = model.evaluate(x_cv, y_cv_cat)
print("Accuracy on CV dataset: {0}".format(acc))

x_len = ImageUtils.lenInPatches(img.shape[1:], (img_dim_y, img_dim_x), 10)
x_all = imgs
y_all = classes
y_all_cat = np_utils.to_categorical(y_all, nb_classes)

res = model.predict(x_all)
img_res = np.argmax(res, axis=1).reshape(res.shape[0] // x_len, x_len)
