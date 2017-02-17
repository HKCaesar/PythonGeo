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
from keras import backend as K

import DataTools
import ImageUtils
import Models

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG, filename='TestUnet.log')

# Network params
class ModelParams(object):
    pass

mp = ModelParams()

mp.nb_classes = 11
mp.nb_epoch = 12

mp.img_dim_x=200
mp.img_dim_y=200
mp.input_shape = (1,mp.img_dim_y,mp.img_dim_x)

mp.epochs = 30
mp.batchSize = 4

modelsPath = join(DataTools.inDir, "models")

model = load_model(join(modelsPath, "gnet_gray_test_6.hdf5"))


imageId = "6100_1_3"

(img, mask) = ImageUtils.loadImage(imageId)

# Predict and save mask
predMask = ImageUtils.getImageMask(img, model, mp, 0.2, 0)
plt.imsave(imageId + ".png", predMask)

# Some visualizations
layer_out = K.function([model.get_layer("input_1").input, K.learning_phase()],
                       [model.get_layer("convolution2d_15").output])
gall = ImageUtils.genPatches(img.shape[1:], (mp.img_dim_y, mp.img_dim_x), 60)
gg = itertools.islice(gall, 20)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gg, img, mask)
layer_out_img = layer_out([imgs, 0])[0]

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
