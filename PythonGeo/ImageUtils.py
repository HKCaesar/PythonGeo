
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

        startX += stride

        if startX + patchW >= imgW:
            startX = 0
            startY += stride

        if startY + patchH >= imgH:
            break

def cutPatch(patch, imageData):
    (startY, startX, patchH, patchW) = patch
    
    maxY = imageData.shape[0] if len(imageData.shape) == 2 else imageData.shape[1]
    maxX = imageData.shape[1] if len(imageData.shape) == 2 else imageData.shape[2]

    assert(startY + patchH < maxY)
    assert(startX + patchW < maxX)

    img = None
    if len(imageData.shape) == 2:
        img = imageData[startY:(startY+patchH), startX:(startX+patchW)]
    else:
        img = imageData[:, startY:(startY+patchH), startX:(startX+patchW)]

    return img


def prepareDataSetFromPatches(patchesGen, imageData, checkFun):
    patchList = list(filter(checkFun, (cutPatch(p, imageData)  for p in patchesGen)))
    n = len(patchList)
    data = np.vstack(patchList)
    return data.reshape((n, data.shape[0]//n) + data.shape[1:])

def prepareDataSets(patchesGen, imageData, mapData):
    patches = [p for p in patchesGen]

    imgList = []
    mapList = []
    mapDetailList = []
    n = 0

    for p in patches:
        patchImg = cutPatch(p, imageData)
        patchMap = cutPatch(p, mapData)

        (patchY, patchX) = patchMap.shape
        patchCentralPoint = patchMap[patchY//2, patchX//2]

        if not patchCentralPoint == 0:
            imgList.append(patchImg)
            mapList.append(patchCentralPoint)
            mapDetailList.append(patchMap)
            n += 1

    stackedImgList = np.stack(imgList)
    #stackedImgList = stackedImgList.reshape((n, stackedImgList.shape[0]//n) + stackedImgList.shape[1:])
    stackedMapList = np.stack(mapList)
    stackedMapDetailList = np.stack(mapDetailList)
    #stackedMapDetailList = stackedMapDetailList.reshape((n, stackedMapDetailList.shape[0]//n) + stackedMapDetailList.shape[1:])

    return (stackedImgList, stackedMapList, stackedMapDetailList)

testId = '6100_1_3'
(img, mask) = DataTools.loadAll(testId)
gall = genPatches(img.shape[1:], (100, 100), 10)
(imgs, classes, masks) = prepareDataSets(gall, img, mask)
