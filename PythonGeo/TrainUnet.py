import SetEnvForGpu

import itertools
import logging
from os.path import join, exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as prc
import sklearn.cross_validation as scv

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

def genPatches(img, mask):
    gall = ImageUtils.genPatches(img.shape[1:], (img_dim_y, img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

def trainOnImage(imageId, model, cbs):

    logging.info("Training on image: {0}".format(imageId))

    (img, mask) = ImageUtils.loadImage(imageId)
    (imgs, classes, masks) = genPatches(img, mask)

    (x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, masks, test_size=0.2)
    y_train_cat = np_utils.to_categorical(y_train.flatten(), nb_classes)
    y_train_cat = y_train_cat.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2], nb_classes))
    
    model.fit(x_train, y_train_cat, nb_epoch=epochs, batch_size=batchSize, callbacks = cbs)

    logging.info("Training completed, evaluating model")

    y_cv_cat = np_utils.to_categorical(y_cv.flatten(), nb_classes)
    y_cv_cat = y_cv_cat.reshape((y_cv.shape[0], y_cv.shape[1]*y_cv.shape[2], nb_classes))

    loss = model.evaluate(x_cv, y_cv_cat, batch_size=batchSize)

    logging.info("Loss: {0}".format(str(loss)))

    return model


model = Models.getGnet(input_shape, nb_classes)

allTrainIds = DataTools.trainImageIds

modelsPath = join(DataTools.inDir, "models")
if not exists(modelsPath):
    makedirs(modelsPath)

checkpointer = ModelCheckpoint(filepath=join(modelsPath, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"), verbose=1, save_best_only=True)
csv_logger = CSVLogger('training.log')

callbacks = [checkpointer, csv_logger]

# Test code - comment out
trainOnImage("6100_1_3", model, callbacks)
model.save(join(modelsPath, "gnet_gray_test.hdf5".format(iteration)))

trainImages = np.random.permutation(allTrainIds)[:7]

logging.info("Training on images: {0}".format(trainImages))

iteration = 0
for imageId in trainImages:
    trainOnImage(imageId, model, callbacks)
    model.save(join(modelsPath, "gnet_gray_it_{0}.hdf5".format(iteration)))
    iteration += 1
