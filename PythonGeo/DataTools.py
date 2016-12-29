
import pandas as pd
import numpy as np

from os.path import join, exists

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
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }
ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
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

def makeClassMask(imageId, nClass):
    (xmax, ymin, W, H) = get_size(imageId)
    polygonList = wkt_loads(df[df.ImageId == imageId].MultipolygonWKT.values[nClass])

    (perimeters, interiors) = getPolygonContours(polygonList, xmax, ymin, W, H)

    mask = plotMask(W, H, (perimeters, interiors), 1)

    return mask
