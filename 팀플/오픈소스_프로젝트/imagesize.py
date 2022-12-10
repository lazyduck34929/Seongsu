import cv2
from PIL import Image
 
imageA = cv2.imread('./image/test1-1.jpg')
imageB = cv2.imread('./image/test1-2.jpg')

print(imageA.shape)
print(imageB.shape)

h, w, c = imageA.shape

imageB = cv2.resize(imageB, (w, h))
print(imageB.shape) 

