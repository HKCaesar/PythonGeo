import SetEnvForGpu

import itertools
import logging
from os.path import join, exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as prc

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger

import DataTools
import ImageUtils
import Models

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG, filename='TrainGen.log')


# Network params
class ModelParams(object):
    pass

mp = ModelParams()

mp.nb_classes = 11

mp.img_dim_x=200
mp.img_dim_y=200
mp.input_shape = (1,mp.img_dim_y,mp.img_dim_x)

mp.epochs = 30
mp.batchSize = 4
mp.samples = 100

def genPatches(img, mask, modelParams):
    gall = ImageUtils.genPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), 60)
    #gg = itertools.islice(gall, nbSamples)
    (imgs, classes, masks) = ImageUtils.prepareDataSets(gall, img, mask)
    return (imgs, classes, masks)

def generateData():
    while True:
        # Choose an image
        idx = np.random.randint(0, len(DataTools.trainImageIds))
        logging.info("Loading image: {0}".format(DataTools.trainImageIds[idx]))
        (img, mask) = ImageUtils.loadImage("6110_3_1") #(DataTools.trainImageIds[idx])

        # Read 100 patches
        (imgs, classes, masks) = genPatches(img, mask, mp)
    
        # Shuffle
        imgsBuff = np.zeros((mp.samples,) + imgs.shape[1:])
        masksBuff = np.zeros((mp.samples,) + masks.shape[1:])

        idxs = np.random.permutation(np.arange(imgs.shape[0]))[:mp.samples]
        logging.info("Subsamples: {0}".format(idxs))
        for i in range(len(idxs)):
            imgsBuff[i] = imgs[idxs[i]]
            masksBuff[i] = masks[idxs[i]]

        x_train = imgsBuff
        y_train_cat = np_utils.to_categorical(masksBuff.flatten(), mp.nb_classes)
        y_train_cat = y_train_cat.reshape((masksBuff.shape[0], masksBuff.shape[1]*masksBuff.shape[2], mp.nb_classes)) # ? Check correctness here ?

        # Feed to network
        for i in range(x_train.shape[0]//mp.batchSize):
            yield (x_train[i*mp.batchSize:(i+1)*mp.batchSize,:], y_train_cat[i*mp.batchSize:(i+1)*mp.batchSize,:])

#xx = [i for i in itertools.islice(generateData(), 1)]

model = Models.getGnet(mp.input_shape, mp.nb_classes)

checkpointer = ModelCheckpoint(filepath="unet_weights.{epoch:02d}.hdf5", verbose=1, save_best_only=True)
csv_logger = CSVLogger('training.log')
callbacks = [checkpointer, csv_logger]

modelsPath = join(DataTools.inDir, "models")
if not exists(modelsPath):
    makedirs(modelsPath)

h = model.fit_generator(generateData(), samples_per_epoch = 2000, nb_epoch = 20, verbose = True, callbacks = callbacks)


model.save(join(modelsPath, "gnet_gen_1.hdf5"))
