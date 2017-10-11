from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import cv2
import numpy as np
import imutils
import string
import random
import glob, os
from os import listdir
from os.path import isfile, join
import skimage.util
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random

# img = cv2.imread('速0.png',0)
# dx = -1+2*np.random.random(size = img.shape);
# dy = -1+2*np.random.random(size = img.shape);
# print (dx)


import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    print (x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


img = cv2.imread('速.png')
img = elastic_transform(img,2,0.4)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# img = cv2.dilate(255 - img, kernel)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


cv2.imshow('test',255 - img)
cv2.waitKey(0)
cv2.destroyAllWindows()