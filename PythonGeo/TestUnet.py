import SetEnvForGpu

import itertools
import logging
from os.path import join, exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as prc
import sklearn.model_selection as scv

from keras.models import load_model

from keras.utils import np_utils

import DataTools
import ImageUtils
import Models

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG, filename='TrainUnet.log')

# Network params
nb_classes = 11
nb_epoch = 12

img_dim_x=200
img_dim_y=200
input_shape = (1,img_dim_y,img_dim_x)

epochs = 20
batchSize = 4

modelsPath = join(DataTools.inDir, "models")

model = load_model(join(modelsPath, "gnet_gray_test_1.hdf5"))

def getImageMask(img, model):
    gall_t = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)
    (imgs_t, classes_t, _) = ImageUtils.prepareDataSets(gall_t, img, np.zeros(img.shape[1:]))
    coords = [x for x in ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), img_dim_x)]
    all_rez = model.predict(imgs_t, batch_size=batchSize)

    rez1 = np.array(all_rez)
    rez1[:,:,0:1] *= 0.1
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, img_dim_y, img_dim_x))

    mask_rez = np.zeros(img.shape[1:])

    for i in range(len(coords)):
        (y, x, h, w) = coords[i]
        mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    return mask_rez

imageId = "6100_1_3"

(img, mask) = ImageUtils.loadImage(imageId)

# Predict and save mask
predMask = getImageMask(img, model)
plt.imsave(imageId + ".png", predMask)

# Evaluate model
def genPatches(img, mask):
    gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

(imgs, classes, masks) = genPatches(img, mask)

x_test = imgs
y_test = masks
y_test_cat = np_utils.to_categorical(y_test.flatten(), nb_classes)
y_test_cat = y_test_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], nb_classes))

model.evaluate(x_test, y_test_cat, batch_size = batchSize)
