import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

from ..algorithms.text_detection import runTD
class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath

        origin = cv2.imread(filepath)
        img = np.copy(origin)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)

        #text detection
        runTD(img)

        #Возможно это оставим
        norm_image = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        print(norm_image)
        cv2.imshow("Image", gray)
        # cv2.imshow("gray", gray)
        # cv2.imshow("mask", mask)
        # cv2.imshow("hsv", hsv)
        cv2.imshow("norm_image", norm_image)
        # cv2.imwrite('./res.jpg', mask)

        # cv2.imshow("binary", binary)




#Тут мы пытались сделать результаты 0 или 1
# image = plt.imread(filepath)
# image.shape
# plt.imshow(image)
#
# gray = rgb2gray(image)
# gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
# for i in range(gray_r.shape[0]):
#     if gray_r[i] > gray_r.mean():
#         gray_r[i] = 1
#     else:
#         gray_r[i] = 0
# gray = gray_r.reshape(gray.shape[0], gray.shape[1])
# plt.imshow(gray, cmap='gray')