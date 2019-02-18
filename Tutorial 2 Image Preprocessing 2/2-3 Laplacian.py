import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("test.jpg", 0)

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap)

plt.imshow(dst,cmap='gray')
plt.show()