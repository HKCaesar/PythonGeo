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
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger

import DataTools
import ImageUtils
import Models

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG, filename='TrainUnet.log')

# Network params
class ModelParams(object):
    pass

mp = ModelParams()

mp.nb_classes = 11

mp.img_dim_x=200
mp.img_dim_y=200
mp.input_shape = (1,mp.img_dim_y,mp.img_dim_x)

mp.epochs = 20
mp.batchSize = 4

def genPatches(img, mask, modelParams):
    gall = ImageUtils.genPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), 60)
    #gg = itertools.islice(gall, 20)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

def trainOnImage(imageId, model, cbs, modelParams):

    logging.info("Training on image: {0}".format(imageId))

    (img, mask) = ImageUtils.loadImage(imageId)
    (imgs, classes, masks) = genPatches(img, mask, modelParams)

    (x_train, x_cv, y_train, y_cv) = scv.train_test_split(imgs, masks, test_size=0.2)
    y_train_cat = np_utils.to_categorical(y_train.flatten(), modelParams.nb_classes)
    y_train_cat = y_train_cat.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2], nmodelParams.b_classes))
    
    model.fit(x_train, y_train_cat, nb_epoch=modelParams.epochs, batch_size=modelParams.batchSize, callbacks = cbs)

    logging.info("Training completed, evaluating model")

    y_cv_cat = np_utils.to_categorical(y_cv.flatten(), modelParams.nb_classes)
    y_cv_cat = y_cv_cat.reshape((y_cv.shape[0], y_cv.shape[1]*y_cv.shape[2], modelParams.nb_classes))

    loss = model.evaluate(x_cv, y_cv_cat, batch_size=modelParams.batchSize)

    logging.info("Loss: {0}".format(str(loss)))

    return model


modelsPath = join(DataTools.inDir, "models")
if not exists(modelsPath):
    makedirs(modelsPath)

#model = Models.getGnet(input_shape, nb_classes)
model = load_model(join(modelsPath, "gnet_gray_test_3.hdf5"))

allTrainIds = DataTools.trainImageIds
trainImages =  ['6110_3_1', '6100_2_3', '6040_1_3', '6010_4_4', '6140_3_1',
       '6110_1_2', '6060_2_3'] # np.random.permutation(allTrainIds)[:7]

# '6040_1_3' - do not use
# '6010_4_4' - do not use


checkpointer = ModelCheckpoint(filepath=join(modelsPath, "weights.{epoch:02d}.hdf5"), verbose=1, save_best_only=True)
csv_logger = CSVLogger('training.log')

callbacks = [checkpointer, csv_logger]

iteration = 0

# Test code - comment out
trainOnImage("6110_1_2", model, callbacks)
model.save(join(modelsPath, "gnet_gray_test_4.hdf5"))

logging.info("Training on images: {0}".format(trainImages))


for imageId in trainImages:
    trainOnImage(imageId, model, callbacks)
    model.save(join(modelsPath, "gnet_gray_it_{0}.hdf5".format(iteration)))
    iteration += 1
