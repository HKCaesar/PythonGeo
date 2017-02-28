#import SetEnvForGpu

from os.path import join, exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import DataTools
import ImageUtils

for imageId in DataTools.trainImageIds:
    (img, mask) = ImageUtils.loadImage(imageId)
    pngDir =  join(DataTools.inDir, "three_band_processed_png")
    if not exists(pngDir):
        makedirs(pngDir)
    plt.imsave("{0}\\{1}.png".format(pngDir, imageId), img.reshape(img.shape[1:]))
