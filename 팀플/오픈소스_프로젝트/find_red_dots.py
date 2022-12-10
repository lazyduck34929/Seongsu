# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import numpy as np

def main():
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

if __name__ == "__main__":
    main()
