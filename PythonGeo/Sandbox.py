#import SetEnvForGpu

from os.path import join, exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

def myToCategorical(masksBuff, nb_classes):
    categoricalTmp = np.zeros((masksBuff.shape[0],) + (masksBuff.shape[1]*masksBuff.shape[2],) + (nb_classes,))
    (batchSize, maskRows, maskCols) = masksBuff.shape
    for b in range(batchSize):
        for c in range(maskCols):
            for r in range(maskRows):
                if masksBuff[b,r,c] > 0:
                    categoricalTmp[b, c + r*maskCols, masksBuff[b,r,c]-1] = 1
    return categoricalTmp

mb = np.array([1,2,3,4,5,6,7,8,9]).reshape((1,3,3))

xx = myToCategorical(mb, 10)