import cv2
import numpy as np
import matplotlib.pyplot as plt

N = 2
gray = cv2.imread('images/ChildBrushingTeeth.jpg',cv2.IMREAD_GRAYSCALE)
Uniform_quant = np.round(gray*(N/255))*np.round(255/N)
print(Uniform_quant)

plt.imshow(Uniform_quant,cmap='gray')
plt.show()

