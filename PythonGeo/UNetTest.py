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

img_dim_x=200
img_dim_y=200
input_shape = (3,img_dim_y,img_dim_x)

model = Models.getGnet(input_shape, nb_classes)

testId = '6100_1_3'
(img_i, mask) = DataTools.loadAll(testId)
img = img_i.astype(float)
img = prc.scale(img.reshape(-1, 1)).reshape(img_i.shape)

gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 60)
gg = itertools.islice(gall, 2000)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)

(x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, masks, test_size=0.2)
y_train_cat = np_utils.to_categorical(y_train.flatten(), nb_classes)
y_train_cat = y_train_cat.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2], nb_classes))

#classWeights = { 0:0.2, 1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.0, 9:1.0, 10:1.0}

# For UNet
#model.fit(x_train, y_train_cat, nb_epoch=15, batch_size=11)

# For GNet
model.fit(x_train, y_train_cat, nb_epoch=10, batch_size=4)

#trainBatchSize = 10
#idx = 0
#ii = 0
#while idx < x_train.shape[0]:
#    print("Training batch: {0}".format(ii))
#    maxIdx = min(idx + trainBatchSize, x_train.shape[0])
#    x_batch = x_train[idx:maxIdx]
#    y_batch = y_train[idx:maxIdx]
#    y_batch_cat = np_utils.to_categorical(y_batch.flatten(), nb_classes)
#    y_batch_cat = y_batch_cat.reshape((y_batch.shape[0], y_batch.shape[1]*y_batch.shape[2], nb_classes))
#    model.train_on_batch(x_batch, y_batch_cat)
#    idx += trainBatchSize
#    ii += 1

y_cv_cat = np_utils.to_categorical(y_cv.flatten(), nb_classes)
y_cv_cat = y_cv_cat.reshape((y_cv.shape[0], y_cv.shape[1]*y_cv.shape[2], nb_classes))

rez = model.predict(x_cv)
rez1 = np.array(rez)
rez1[:,:,0:1] *= 0.1
#rez1[:,:,0:1] = 0
xx = np.argmax(rez1, axis = 2)
xx = xx.reshape((-1, img_dim_y, img_dim_x))


gall_t = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)
(imgs_t, classes_t, masks_t) = ImageUtils.prepareDataSets(gall_t, img, mask)
coords = [x for x in ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)]
all_rez = model.predict(imgs_t, batch_size=4)
rez1 = np.array(all_rez)
rez1[:,:,0:1] *= 0.1
xx = np.argmax(rez1, axis = 2)
xx = xx.reshape((-1, img_dim_y, img_dim_x))

mask_rez = np.zeros(img.shape[1:])

for i in range(len(coords)):
    (y, x, h, w) = coords[i]
    mask_rez[y:(y+h), x:(x+h)] = xx[i]

def getImageMask(img, model):
    gall_t = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)
    (imgs_t, classes_t, masks_t) = ImageUtils.prepareDataSets(gall_t, img, mask)
    coords = [x for x in ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)]
    all_rez = model.predict(imgs_t, batch_size=4)

    rez1 = np.array(all_rez)
    rez1[:,:,0:1] *= 0.1
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, img_dim_y, img_dim_x))

    mask_rez = np.zeros(img.shape[1:])

    for i in range(len(coords)):
        (y, x, h, w) = coords[i]
        mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    return mask_rez
