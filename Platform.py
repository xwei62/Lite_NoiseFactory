import cv2
import numpy as np

img = cv2.imread('速0.png',0)

size = np.size(img)
skel = np.zeros(img.shape,np.uint8)
#
# ret,img = cv2.threshold(img,127,255,0)
# img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY, 11, 5)
# img = cv2.resize(img,(55,55),interpolation=cv2.INTER_AREA)
ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False

while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()