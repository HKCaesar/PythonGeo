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
class ModelParams(object):
    pass

mp = ModelParams()

mp.nb_classes = 11
mp.nb_epoch = 12

mp.img_dim_x=200
mp.img_dim_y=200
mp.input_shape = (1,mp.img_dim_y,mp.img_dim_x)

mp.epochs = 20
mp.batchSize = 4

modelsPath = join(DataTools.inDir, "models")

model = load_model(join(modelsPath, "gnet_gray_test_4.hdf5"))

def getImageMask(img, model, modelParams, backgoungPenality, border):
    gall_t = ImageUtils.genPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x-border)
    (imgs_t, classes_t, _) = ImageUtils.prepareDataSets(gall_t, img, np.zeros(img.shape[1:]))
    coords = [x for x in ImageUtils.genPatches(img.shape[1:], (modelParams.img_dim_y-2*border, modelParams.img_dim_x-2*border),
                                               modelParams.img_dim_x-2*border)]
    all_rez = model.predict(imgs_t, batch_size=modelParams.batchSize)

    rez1 = np.array(all_rez)
    rez1[:,:,0:1] *= backgoungPenality
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, modelParams.img_dim_y, modelParams.img_dim_x))

    mask_rez = np.zeros(img.shape[1:])

    for i in range(len(coords)):
        (y, x, h, w) = coords[i]
        if border != 0:
            mask_rez[y:(y+h), x:(x+h)] = rez_img[i][border:-border,border:-border]
        else:
            mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    # Fill borders
    borderPatches = ImageUtils.genBorderPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x)
    (imgs_b, classes_b, _) = ImageUtils.prepareDataSets(borderPatches, img, np.zeros(img.shape[1:]))
    borderCoords = [x for x in ImageUtils.genBorderPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x)]

    borderRez = model.predict(imgs_b, batch_size=modelParams.batchSize)
    rez1 = np.array(borderRez)
    rez1[:,:,0:1] *= backgoungPenality
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, modelParams.img_dim_y, modelParams.img_dim_x))

    for i in range(len(borderCoords)):
        (y, x, h, w) = borderCoords[i]
        mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    return mask_rez

imageId = "6100_1_3"

(img, mask) = ImageUtils.loadImage(imageId)

# Predict and save mask
predMask = getImageMask(img, model, mp, 0.2, 5)
plt.imsave(imageId + ".png", predMask)

# Evaluate model
def genPatches(img, mask):
    gall = ImageUtils.genPatches(img.shape[1:], (mp.img_dim_y, mp.img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

(imgs, classes, masks) = genPatches(img, mask)

x_test = imgs
y_test = masks
y_test_cat = np_utils.to_categorical(y_test.flatten(), mp.nb_classes)
y_test_cat = y_test_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], mp.nb_classes))

model.evaluate(x_test, y_test_cat, batch_size = mp.batchSize)
