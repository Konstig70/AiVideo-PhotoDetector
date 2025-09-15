import cv2
import numpy as np

img = np.zeros((100,100,3), dtype=np.uint8)
cv2.imshow('test', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
