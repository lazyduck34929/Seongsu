import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

#성수
from skimage.metrics import structural_similarity as compare_ssim
import imutils

def imagesize():
  imageA = cv2.imread('./image/test1-1.jpg')
  imageB = cv2.imread('./image/test1-2.jpg')

  print(imageA.shape)
  print(imageB.shape)

  h, w, c = imageA.shape

  imageB = cv2.resize(imageB, (w, h))
  print(imageB.shape) 
  
def red_dot():
  imageA = cv2.imread('./image/original.jpg')
  imageB = cv2.imread('./image/copy.jpg')
  imageC = imageA.copy()

  tempDiff = cv2.subtract(imageA, imageB)
    
  grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
  (score, diff) = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
    
  print("Similarity:", score)
    
  thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  tempDiff[thresh == 255] = [0, 0, 255]
  imageC[thresh == 255] = [0, 0, 255]

  cv2.imshow("compare", imageC)
  cv2.waitKey(0)

def section():
  imageA = cv2.imread('./image/original.jpg')
  imageB = cv2.imread('./image/copy.jpg')
  
  grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
  (score, diff) = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  print(f"SSIM: {score}")
  thresh = cv2.threshold(
               diff, 0, 200, 
               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
           )[1]
  cnts, _ = cv2.findContours(
              thresh, 
              cv2.RETR_EXTERNAL, 
              cv2.CHAIN_APPROX_SIMPLE
            )
  for c in cnts:
      area = cv2.contourArea(c)
      if area > 40:
          x, y, w, h = cv2.boundingRect(c)
          cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
          cv2.drawContours(imageB, [c], -1, (0, 0, 255), 2)
  cv2.imshow("Original", imageA)
  cv2.waitKey(0)

  
#다른 분
