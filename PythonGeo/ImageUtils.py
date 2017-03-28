
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn.preprocessing as prc

import DataTools

def showImg(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

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

def genBorderPatches(imgShape, patchShape, stride):
    (imgH, imgW) = imgShape
    (patchH, patchW) = patchShape


    # Yield right-side patches
    (startY, startX) = (0, imgW - patchW -1)

    for i in range(imgH // patchH):
        patch = (startY + patchH*i, startX, patchH, patchW)
        yield patch

    # Yield bottom-side patches
    (startY, startX) = (imgH - patchH - 1, 0)

    cnt = imgW // patchW if imgH % patchH != 0 else (imgW // patchW) - 1 # Avoid doubling of bottom-right patch

    for i in range(cnt):
        patch = (startY, startX + patchW*i, patchH, patchW)
        yield patch

    # Yield bottom-right patch
    if imgH % patchH != 0 and imgW % patchW != 0:
        patch = (imgH - patchH - 1, imgW - patchW - 1, patchH, patchW)
        yield patch

def lenInPatches(imgShape, patchShape, stride):
    (imgH, imgW) = imgShape
    (patchH, patchW) = patchShape
    (startY, startX) = (0, 0)

    count = 0
    while True:

        startX += stride
        count += 1

        if startX + patchW >= imgW:
            return count

    return count


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

        #if not patchCentralPoint == 0:
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

def loadImage(imageId):
    (img_i, mask) = DataTools.loadAll(imageId)
    img = np.mean(scale_percentile(img_i), axis=0) #img_i.astype(float)
    img = prc.scale(img.reshape(-1, 1)).reshape((1,) + img.shape)
    return (img, mask)

def getRawPredictions(img, model, modelParams, border):
    gall_t = genPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x-border)
    (imgs_t, classes_t, _) = prepareDataSets(gall_t, img, np.zeros(img.shape[1:]))
    coords = [x for x in genPatches(img.shape[1:], (modelParams.img_dim_y-2*border, modelParams.img_dim_x-2*border),
                                               modelParams.img_dim_x-2*border)]
    all_rez = model.predict(imgs_t, batch_size=modelParams.batchSize)
    return (all_rez,coords)

def getImageMask(img, model, modelParams, border):
    (all_rez,coords) = getRawPredictions(img, model, modelParams, border)

    rez1 = np.array(all_rez)
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, modelParams.img_dim_y, modelParams.img_dim_x))

    mask_rez = np.zeros(img.shape[1:])

    for i in range(len(coords)):
        (y, x, h, w) = coords[i]
        if border != 0:
            mask_rez[y:(y+h), x:(x+h)] = rez_img[i][border:-border,border:-border]
        else:
            mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    # Fill borders
    borderPatches = genBorderPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x)
    (imgs_b, classes_b, _) = prepareDataSets(borderPatches, img, np.zeros(img.shape[1:]))
    borderCoords = [x for x in genBorderPatches(img.shape[1:], (modelParams.img_dim_y, modelParams.img_dim_x), modelParams.img_dim_x)]

    borderRez = model.predict(imgs_b, batch_size=modelParams.batchSize)
    rez1 = np.array(borderRez)
    rez_img = np.argmax(rez1, axis = 2)
    rez_img = rez_img.reshape((-1, modelParams.img_dim_y, modelParams.img_dim_x))

    for i in range(len(borderCoords)):
        (y, x, h, w) = borderCoords[i]
        mask_rez[y:(y+h), x:(x+h)] = rez_img[i]

    return mask_rez


#testId = '6100_1_3'
#(img, mask) = DataTools.loadAll(testId)
#gall = genPatches(img.shape[1:], (100, 100), 10)
#(imgs, classes, masks) = prepareDataSets(gall, img, mask)
