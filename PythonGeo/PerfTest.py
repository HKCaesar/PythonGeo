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

# Evaluate model
def genPatches(img, mask, mp):
    gall = ImageUtils.genPatches(img.shape[1:], (mp.img_dim_y, mp.img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

def evaluateOnImage(model, mp, imageId):
    (img, mask) = ImageUtils.loadImage(imageId)
    (imgs, classes, masks) = genPatches(img, mask, mp)
    x_test = imgs
    y_test = masks
    y_test_cat = DataTools.myToCategorical(y_test.flatten(), mp.nb_classes)
    y_test_cat = y_test_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], mp.nb_classes))

    rez = model.evaluate(x_test, y_test_cat, batch_size = mp.batchSize)

    del img
    del mask
    del imgs
    del classes
    del masks
    del y_test_cat

    return rez

def evalOnList(model, mp, images):

    resMap = {}

    for i in images:
        logging.info("Evaluating on image: {0}".format(i))
        (loss, acc) = evaluateOnImage(model, mp, i)
        logging.info("Evaluation results: loss {0}, acc {1}".format(loss, acc))
        resMap[i] = (loss, acc)

    return resMap

def saveMap(filePath, map):
    with open(filePath, 'w') as f:
        f.write("imageId,loss,acc\n")
        for imageId in map.keys():
            (loss, acc) = map[imageId]
            f.write("{0},{1},{2}\n".format(imageId, loss, acc))

modelsToTest = [
    "gnet_gen_f_5",
    "gnet_gen_f_6"
    ]

for modelFileName in modelsToTest:
    model = load_model(join(modelsPath, modelFileName + ".hdf5"))
    logging.info("Loaded model: {0}".format(modelFileName))
    map = evalOnList(model, mp, DataTools.trainImageIds)
    del model
    saveMap(join(modelsPath, modelFileName + "_eval.csv"), map)
