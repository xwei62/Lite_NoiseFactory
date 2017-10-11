
# import the necessary packages
import numpy as np

import cv2

img = cv2.imread('内蒙古.jpg',0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,4))
img = cv2.erode(255 - img,kernel)
img = cv2.minAreaRect(img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
