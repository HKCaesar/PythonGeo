
import pandas as pd
import numpy as np

from os.path import join, exists
from os import makedirs

from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch

import matplotlib.pyplot as plt
import tifffile as tiff

import pylab

import cv2

# turn interactive mode on so that plots immediately
# See: http://stackoverflow.com/questions/2130913/no-plot-window-in-matplotlib
# pylab.ion()

inDir = '..\\..\\Data\\'

# Give short names, sensible colors and zorders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }

ZORDER = {
        1 : 8,
        2 : 7,
        3 : 6,
        4 : 5,
        5 : 4,
        6 : 3,
        7 : 1,
        8 : 2,
        9 : 9,
        10: 10,
        }

# read the training data from train_wkt_v4.csv
df = pd.read_csv(join(inDir, 'train_wkt_v4.csv'))

# grid size will also be needed later..
gs = pd.read_csv(join(inDir, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

# imageIds in a DataFrame
allImageIds = gs.ImageId.unique()
trainImageIds = df.ImageId.unique()

def get_image_names(imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': '{}three_band/{}.tif'.format(inDir, imageId),
         'A': '{}/sixteen_band/{}_A.tif'.format(inDir, imageId),
         'M': '{}/sixteen_band/{}_M.tif'.format(inDir, imageId),
         'P': '{}/sixteen_band/{}_P.tif'.format(inDir, imageId),
         }
    return d


def get_images(imageId, img_key = None):
    '''
    Load images correspoding to imageId

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    img_key : {None, '3', 'A', 'M', 'P'}, optional
        Specify this to load single image
        None loads all images and returns in a dict
        '3' loads image from three_band/
        'A' loads '_A' image from sixteen_band/
        'M' loads '_M' image from sixteen_band/
        'P' loads '_P' image from sixteen_band/

    Returns
    -------
    images : dict
        A dict of image data from TIFF files as numpy array
    '''
    img_names = get_image_names(imageId)
    images = dict()
    if img_key is None:
        for k in img_names.keys():
            images[k] = tiff.imread(img_names[k])
    else:
        images[img_key] = tiff.imread(img_names[img_key])
    return images

def get_size(imageId):
    """
    Get the grid size of the image

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    """
    xmax, ymin = gs[gs.ImageId == imageId].iloc[0,1:].astype(float)
    W, H = get_images(imageId, '3')['3'].shape[1:]
    return (xmax, ymin, W, H)

def is_training_image(imageId):
    '''
    Returns
    -------
    is_training_image : bool
        True if imageId belongs to training data
    '''
    return any(trainImageIds == imageId)


def coordsToRaster(coords, xmax, ymin, W, H):
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/xmax
    yf = H1/ymin
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def getPolygonContours(polygonList, xmax, ymin, W, H):
    perim_list = []
    interior_list = []

    for k in range(len(polygonList)):
        currPoly = polygonList[k]
        perim = np.array(list(currPoly.exterior.coords))
        perim_c = coordsToRaster(perim, xmax, ymin, W, H)
        perim_list.append(perim_c)
        for pi in currPoly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = coordsToRaster(interior, xmax, ymin, W, H)
            interior_list.append(interior_c)

    return perim_list, interior_list

def plotMask(W, H, contours, fillValue):
    img_mask = np.zeros((H, W), np.uint8)
    if contours is None:
        return img_mask

    (perimeters, interiors) = contours
    cv2.fillPoly(img_mask, perimeters, fillValue)
    cv2.fillPoly(img_mask, interiors, 0)

    return img_mask

def plotPolygons(imageId, nClass, xmax, ymin, W, H, fillValue):
    polygonList = wkt_loads(df[(df.ImageId == imageId) & (df.ClassType == nClass)].MultipolygonWKT.values[0])

    (perimeters, interiors) = getPolygonContours(polygonList, xmax, ymin, W, H)

    mask = plotMask(W, H, (perimeters, interiors), fillValue)

    return mask

def makeClassMask(imageId, nClass, fillValue):
    (xmax, ymin, W, H) = get_size(imageId)
    
    return plotPolygons(imageId, nClass, xmax, ymin, W, H, fillValue)

def makeCombinedMask(imageId):
    (xmax, ymin, W, H) = get_size(imageId)

    layers = np.zeros((H, W, len(CLASSES)))

    for nClass in CLASSES.keys():
        layers[:,:,(nClass-1)] = makeClassMask(imageId, nClass, ZORDER[nClass])

    combinedMask = np.amax(layers, axis=2).astype(np.int8)

    return combinedMask

processedDir = join(inDir, "three_band_processed")
def processImage(imageId):
    combinedMask = makeCombinedMask(imageId)
    destDir = processedDir
    if not exists(destDir):
        makedirs(destDir)

    destFileNpy = join(processedDir, "{0}.npy".format(imageId))
    destFilePng = join(processedDir, "{0}.png".format(imageId))
    np.save(destFileNpy, combinedMask)
    plt.imsave(destFilePng, combinedMask)

def loadAll(imageId):
    npyFile = join(processedDir, "{0}.npy".format(imageId))
    if not exists(npyFile):
        processImage(imageId)
    mask = np.load(npyFile)

    rawImage = get_images(imageId, '3')['3']
    axesCorrectedImage = rawImage #np.transpose(rawImage, (1, 2, 0))

    return (axesCorrectedImage, mask)

# Experiment
#testId = '6100_1_3'
#df[(df.ImageId == testId) & (df.ClassType == 1)]
#class1Mask = makeClassMask(testId, 1, 1)
#class2Mask = makeClassMask(testId, 2, 2)

#cmb = np.zeros(class1Mask.shape + (2,))
#cmb[:,:,0] = class1Mask
#cmb[:,:,1] = class2Mask

#cm = makeCombinedMask(testId)

#(img, mask) = loadAll(testId)
