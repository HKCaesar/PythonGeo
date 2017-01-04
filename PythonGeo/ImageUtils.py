
import numpy as np
import matplotlib.pyplot as plt
import itertools

import DataTools

def genNums(x):
    i = 0
    while i < x:
        yield i
        i = i + 1

def genPatches(imgShape, patchShape, stride):
    (imgH, imgW) = imgShape
    (patchH, patchW) = patchShape

    (startY, startX) = (0, 0)

    while True:
        patch = (startY, startX, patchH, patchW)
        yield patch

        startX = startX + stride

        if startX + patchW > imgW:
            startX = 0
            startY = startY + stride

        if startY + patchH > imgH:
            break

def cutPatch(patch, imageData):
    (startY, startX, patchH, patchW) = patch
    return imageData[startY:(startY+patchH), startX:(startX+patchW)]

def flattenImg(img):
    return img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

def prepareDataSetFromPatches(patchesGen, imageData):
    patchList = [flattenImg(cutPatch(p, imageData))  for p in patchesGen]
    n = len(patchList)
    data = np.vstack(patchList)
    return data.reshape((n, data.shape[0]//n, data.shape[1]))

