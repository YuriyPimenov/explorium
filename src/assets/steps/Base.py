import numpy as np
import cv2
import imutils
from skimage import measure
from pythonRLSA import rlsa
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

from ..algorithms.utils import Average
class Base:
    def __init__(self, debug=True):
        self.debug = debug

    def show(self, name = 'base', img=None, debug=None):
        if debug == None:
            debug = self.debug

        if debug == True:
            cv2.imshow(name, img)

    def showHist(self, dataNames, dataValues, debug=None):
        if debug == None:
            debug = self.debug

        if debug == True:
            plt.figure(figsize=(19, 5))

            # plt.subplot(131)
            plt.bar(dataNames, dataValues)
            # plt.subplot(132)
            # plt.scatter(dataNames, dataValues)
            # plt.subplot(133)
            # plt.plot(dataNames, dataValues)
            plt.suptitle('Analyze connected components')
            plt.xlabel('labels')
            plt.ylabel('count pixels')
            plt.xticks(dataNames, tuple(dataNames), rotation=-90)
            plt.grid(True)
            plt.show()

    def removeLittlePieces(self, img):
        # Начинаем делать анализ соединенных компонентов
        labels = measure.label(img, neighbors=8, background=0)
        mask = np.zeros(img.shape, dtype="uint8")

        dataNames = []
        dataValues = []
        masksLabel = []
        masksHeights = []
        masksWidths = []
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(img.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels == 0:
                continue
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"

            # if numPixels > 300:
            #     mask = cv2.add(mask, labelMask)

            width, height = self.getHeightAndWidthComp(labelMask)

            if width > 0:
                masksWidths.append(width)

            if height > 0:
                masksHeights.append(height)

            masksLabel.append(labelMask)
            dataNames.append(label)
            dataValues.append(numPixels)

        # Получаем среднюю высоту
        avgHeight = Average(masksHeights)
        avgWidth = Average(masksWidths)
        avgCountPixels = Average(dataValues)
        N = 5.0
        T1 = N * avgCountPixels
        # T1 = N * max(dataValues)
        T2 = max(avgWidth, avgHeight)
        r = range(int(1 / T2), int(T2))
        for i in range(len(masksLabel)):
            area = cv2.countNonZero(masksLabel[i])
            width, height = self.getHeightAndWidthComp(masksLabel[i])
            if area > T1:
                hw = height / width

                # if hw < 1/T2 or hw > T2:
                if height >= math.sqrt(T1) and width >= math.sqrt(T1):
                    mask = cv2.add(mask, masksLabel[i])
                    self.show(str(i), masksLabel[i])
        return mask, dataNames, dataValues
