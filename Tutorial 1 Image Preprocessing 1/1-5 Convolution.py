import cv2
from scipy import signal
import matplotlib.pyplot as plt

img = cv2.imread('images/empire.jpg',cv2.IMREAD_GRAYSCALE)
kernel = [[1,1,1],[0,0,0],[-1,-1,-1]]
dest = signal.convolve2d(img,kernel)
plt.imshow(dest,cmap='gray')
plt.show()