import cv2
import numpy as np
img = cv2.imread('sudoku.png',cv2.IMREAD_COLOR)
(height,width,deep) = img.shape
# lefttop, letbottom,rightbuttom
matSrc = np.float32([[505,25],[110,530],[1025,390]])
matDest = np.float32([[0,0],[0, 970],[1100,0]])
matAffine = cv2.getAffineTransform(matSrc,matDest)
dest_img = cv2.warpAffine(img,matAffine,(0,0))
cv2.imshow('img',dest_img)
cv2.waitKey(0)

