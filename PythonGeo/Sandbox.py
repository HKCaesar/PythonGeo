import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

p_0_0 = tiff.imread("..\\..\\Data\\three_band\\6100_1_3.tif")
p_0_1 = tiff.imread("..\\..\\Data\\three_band\\6010_0_1.tif")

plt.imshow(np.transpose(p_0_0, (1, 2, 0)), cmap = 'coolwarm')

data = numpy.swapaxes(p_0_0, -3, -2)
data = numpy.swapaxes(data, -2, -1)

figure, subplot, image = tiff.imshow(p_0_0)

fig = plt.figure()
ax = fig.add_subplot(1,2,1) # two rows, one column, first plot
plt.imshow(np.transpose(p_0_0, (1, 2, 0)))
ax=fig.add_subplot(1,2,2)
plt.imshow(np.transpose(p_0_1, (1, 2, 0)))

p = np.hstack((np.transpose(p_0_0, (1, 2, 0)), np.transpose(p_0_1, (1, 2, 0))))
plt.imshow(p)
tiff.imshow(p)

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

import DataTools

testId = '6100_1_3'
(img, mask) = DataTools.loadAll(testId)

gall = genPatches(img.shape[:2], (100, 100), 10)
gg = itertools.islice(gall, 20)
ds = prepareDataSetFromPatches(gg, img)

gall = genPatches(img.shape[:2], (100, 100), 10)
gg = itertools.islice(gall, 20)
ms = prepareDataSetFromPatches(gg, mask.reshape(mask.shape + (1,)))
