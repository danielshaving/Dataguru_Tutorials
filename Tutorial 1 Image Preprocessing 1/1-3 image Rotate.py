import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('ChildBrushingTeeth.jpg',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
(height,width,deep) = img.shape
matRotation = np.array(
 [[0.5,0.3,0],
  [0.2, 0.7, 0]])
dest_img= cv2.warpAffine(img,matRotation,(0,0))
plt.imshow(dest_img)
plt.figure(1)
plt.show()

