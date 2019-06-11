import numpy as np
import cv2

class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath

        origin = cv2.imread(filepath)
        img = np.copy(origin)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        # _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # binary = cv2.cvtColor(mask, cv2.CV_32F)
        # dst = mask.astype(np.float32)
        hsv = np.asarray(mask, dtype=np.float32)
        print(mask)
        cv2.imshow("Image", img)
        cv2.imshow("gray", gray)
        cv2.imshow("mask", mask)
        cv2.imshow("hsv", hsv)
        # cv2.imshow("binary", binary)


