import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage.metrics import structural_similarity as compare_ssim
import imutils
from PIL import Image
from PIL import ImageChops

#-------------------------function by yushin-------------------------
def Cut_Image_Circle(img):
    #출처: https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0

    #1. Read Image
    img = cv2.imread(image)

    #2. Make a mask
    mask = np.zeros_like(img)
    #np.zeros_like(varialbe) 어떤 변수의 사이즈만큼 0으로 가득찬 array를 만든다.
    cv2.circle(mask, (260,210), 100, (255,255,255), -1)
    #cv2.circle(대상이미지, (원점x, 원점y), 반지름, (색상), 채우기)

    #Masking
    masked = cv2.bitwise_and(img, mask)
    #bit 연산 and (논리곱)

    #Result Print
    cv2.imshow('original', img)
    cv2.imshow('mask', mask)
    cv2.imshow('masked', masked)
    cv2.waitKey()
    #cv2.waitKey(): 
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows(): 



def Diff_Image(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #두 영상의 절대값 차 연산
    diff = cv2.absdiff(img1_gray, img2_gray)

    #차 영상을 극대화 하기 위해 쓰레시홀드 처리 및 컬러로 변환
    _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    diff_red[:, :, 2] = 0

    #두 번째 이미지에 변화 부분 표시
    spot = cv2.bitwise_xor(img2, diff_red)

    #⑤ 결과 영상 출력
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('diff', diff)
    cv2.imshow('spot', spot)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
#-------------------------function by seongsu-------------------------
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



##----------------------main section-------------------------

image = "./image/test_image1.jpg"
Cut_Image_Circle(image)


image1 = "./image/test_image3.jpg"
image2 = "./image/test_image3.jpg"
Diff_Image(image1, image2)
