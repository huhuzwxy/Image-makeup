import cv2
import numpy as np
from math import *

image = cv2.imread('4.jpg')
row, col, channel = image.shape
cv2.namedWindow('原图')
cv2.imshow('原图', image)
cv2.waitKey(0)

#image_crop = image[20:350, 150:550]
#cv2.namedWindow('裁剪')
#cv2.imshow('裁剪', image_crop)
#cv2.waitKey(0)

#image_flip = cv2.flip(image_crop,1)
#cv2.namedWindow('左右镜像')
#cv2.imshow('镜像', image_flip)
#cv2.waitKey(0)

#image_resize = cv2.resize(image_flip,(256,256))
#cv2.namedWindow('缩放')
#cv2.imshow('缩放',image_resize)
#cv2.waitKey(0)

#row, col, channel = image_resize.shape
M = cv2.getRotationMatrix2D((row/2, col/2), 45, 1)
#cos = np.abs(M[0, 0])
#sin = np.abs(M[0, 1])
#nW = int((row * sin) + (col * cos))
#nH = int((row * cos) + (col * sin))
#M[0, 2] += (nW / 2) - row/2
#M[1, 2] += (nH / 2) - col/2
image_rotate = cv2.warpAffine(image, M, (col, row))
cv2.imwrite('rotate1.jpg',image_rotate)
cv2.namedWindow('旋转')
cv2.imshow('旋转', image_rotate)
cv2.waitKey(0)

M1 = cv2.getRotationMatrix2D((row/2, col/2), 45, 1)
row_new = int(col * fabs(sin(radians(45))) + row * fabs(cos(radians(45))))
col_new = int(row * fabs(sin(radians(45))) + col * fabs(cos(radians(45))))
M1[0, 2] += (col_new - col)/2
M1[1, 2] += (row_new - row)/2
image_rotate1 = cv2.warpAffine(image, M1, (col_new, row_new))
cv2.imwrite('rotate2.jpg',image_rotate1)
cv2.namedWindow('旋转')
cv2.imshow('旋转', image_rotate1)
cv2.waitKey(0)