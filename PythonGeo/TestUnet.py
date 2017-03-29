import SetEnvForGpu

import itertools
import logging
from os.path import join, exists
from os import makedirs
import collections

import numpy as np
import matplotlib.pyplot as plt
import cv2

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

mp.nb_classes = 10
mp.nb_epoch = 12

mp.img_dim_x=200
mp.img_dim_y=200
mp.input_shape = (1,mp.img_dim_y,mp.img_dim_x)

mp.epochs = 30
mp.batchSize = 4

modelsPath = join(DataTools.inDir, "models")

modelFileName = "gnet_newcat_2"
model = load_model(join(modelsPath, modelFileName + ".hdf5"))


imageId = "6110_1_2"

(img, mask) = ImageUtils.loadImage(imageId)

img_blur = cv2.GaussianBlur(img.reshape(img.shape[1:]), (5,5), 0).reshape(img.shape)
#img_blur = cv2.bilateralFilter(img.reshape(img.shape[1:]), 3, 5, 10).reshape(img.shape)

rawPred = ImageUtils.getRawPredictions(img, model, mp, 0)

# Predict and save mask
predMask = ImageUtils.getImageMask(img, model, mp, 0)
plt.imsave(imageId + ".png", predMask)

predMask_blur = ImageUtils.getImageMask(img_blur, model, mp, 0)
plt.imsave(imageId + "_blur.png", predMask_blur)

def plotClasses(predMask, mask, modelParams):
    # Count predictions
    cc = collections.Counter(predMask.flatten().astype(int))
    ccm = collections.Counter(mask.flatten())

    ind = np.arange(modelParams.nb_classes)     # the x locations for the groups
    width = 0.4                                # the width of the bars

    fig, ax = plt.subplots()
    predVals = [cc[i] for i in ind]
    rects1 = ax.bar(ind, predVals, width, color='r')
    maskVals = [ccm[i] for i in ind]
    rects2 = ax.bar(ind + width, maskVals, width, color='y')

    ax.set_title('Classes in predicted vs ground truth')
    ax.legend((rects1[0], rects2[0]), ('Predicted', 'Baseline'))

    plt.show()

plotClasses(predMask, mask, mp)
plotClasses(predMask_blur, mask, mp)

# Some visualizations
layer_out = K.function([model.get_layer("input_1").input, K.learning_phase()],
                       [model.get_layer("convolution2d_7").output])
gall = ImageUtils.genPatches(img.shape[1:], (mp.img_dim_y, mp.img_dim_x), 60)
gg = itertools.islice(gall, 20)
(imgs, classes, masks) = ImageUtils.prepareDataSets(gg, img, mask)
layer_out_img = layer_out([imgs, 0])[0]

# Evaluate model
def genPatches(img, mask, mp):
    gall = ImageUtils.genPatches(img.shape[1:], (mp.img_dim_y, mp.img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

#(imgs, classes, masks) = genPatches(img, mask, mp)

#x_test = imgs
#y_test = masks
#y_test_cat = np_utils.to_categorical(y_test.flatten(), mp.nb_classes)
#y_test_cat = y_test_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], mp.nb_classes))

#model.evaluate(x_test, y_test_cat, batch_size = mp.batchSize)

def evaluateLoaded(model, mp, img, mask):
    (imgs, classes, masks) = genPatches(img, mask, mp)

    x_test = imgs
    y_test = masks
    y_test_cat = np_utils.to_categorical(y_test.flatten(), mp.nb_classes)
    y_test_cat = y_test_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], mp.nb_classes))

    return model.evaluate(x_test, y_test_cat, batch_size = mp.batchSize)

evalRez = evaluateLoaded(model, mp, img_blur, mask)

def evaluateOnImage(model, mp, imageId):
    (img, mask) = ImageUtils.loadImage(imageId)
    (imgs, classes, masks) = genPatches(img, mask, mp)
    x_test = imgs
    y_test = masks
    y_test_cat = np_utils.to_categorical(y_test.flatten(), mp.nb_classes)
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
    "gnet_gray_test_1",
    "gnet_gray_test_2",
    "gnet_gray_test_3",
    "gnet_gray_test_4",
    "gnet_gray_test_5",
    "gnet_gray_test_6"
    ]

for modelFileName in modelsToTest[:1]:
    model = load_model(join(modelsPath, modelFileName + ".hdf5"))
    logging.info("Loaded model: {0}".format(modelFileName))
    map = evalOnList(model, mp, DataTools.trainImageIds[:3])
    saveMap(join(modelsPath, modelFileName + "_eval.csv"), map)
