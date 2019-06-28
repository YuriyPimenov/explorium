import numpy as np
import cv2
import math
import matplotlib as plt
from skimage import measure
from ..algorithms.utils import Average
from .Base import Base
class SliceThicknessFilter(Base):
    def __init__(self, A, Ms, row, col, mask, debug=True):
        super().__init__(debug)

        self.A = A
        self.mask = mask
        self.show('step3-start', self.mask)
        self.Ms = Ms
        self.Mt = np.zeros((row, col), np.uint8)

        self.row = row
        self.col = col
        self.test()

    def test(self):
        row = self.row
        col = self.col
        data = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10:0
        }
        testImg = np.zeros((row, col, 3), np.uint8)
        for rowEl in range(row):
            for colEl in range(col):
                element = self.Ms[rowEl][colEl]
                ph, pv, pd, pe = element
                if ph!=0 and pv!=0 and pd!=0 and pe!=0:
                    slices = sorted(element)
                    if slices[0]==slices[1] or (slices[0]>1 and slices[0]*2<slices[1]):
                        self.Mt[rowEl][colEl] = slices[0]
                    else:
                        self.Mt[rowEl][colEl] = Average([slices[0], slices[1]])

                    self.setTestImg(self.Mt, testImg, rowEl, colEl, data)
                    print('{0}-{1}'.format(rowEl, colEl))
        print(self.Mt)
        # self.show('Input', testImg)

        # dataNames = [k for k in data]
        # dataValues = [v for v in data.values()]
        # self.showHist(dataNames, dataValues, True)

        #most common slice thickness
        Tc = max(data, key=data.get)
        for rowEl in range(row):
            for colEl in range(col):
                element = self.Mt[rowEl][colEl]
                if element==1 or element>2*Tc:
                    self.A[rowEl][colEl] = 0.0
                    self.mask[rowEl][colEl] = 0
        self.show('step3-finish', self.mask)

        A_not = cv2.bitwise_not(self.mask)
        A_not = self.mask

        self.show('step3-finish2', A_not)
        # Удаляем маленькие кусочки и текст
        mask, dataNames, dataValues = self.removeLittlePieces(A_not)


        result1 = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.mask = mask
        self.result1 = result1
        self.show('mask', mask)



    def setTestImg(self, Mt, testImg, rowEl, colEl, data):
        thickness = {
            1: [0, 0, 255],#red
            2: [0, 255, 0],#Lime
            3: [255, 0, 0],#Blue
            4: [0, 255, 255],#yellow
            5: [255, 0, 255],#Fuchsia
            6: [147, 20, 255],#pink
            7: [139, 139, 0],#DarkCyan
            8: [0, 69, 255],#orange
            9: [105, 105, 105]#DimGray
        }
        key = Mt[rowEl][colEl]
        try:
            thick = thickness[key]
        except KeyError as e:
            thick = [255, 255, 255]
            # можно также присвоить значение по умолчанию вместо бросания исключения
            # raise ValueError('Undefined unit: {}'.format(e.args[0]))
        testImg[rowEl][colEl] = thick

        if key > 9:
            data[10] = data[10] + 1
        else:
            data[key] = data[key]+1

    def getHeightAndWidthComp(self, comp):
        leftmost = comp.shape[1]
        rightmost = 0
        topmost = comp.shape[0]
        bottommost = 0
        contours, hierarchy = cv2.findContours(comp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            leftmostTMP = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmostTMP = tuple(contour[contour[:, :, 0].argmax()][0])
            topmostTMP = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommostTMP = tuple(contour[contour[:, :, 1].argmax()][0])

            if leftmostTMP[0] < leftmost:
                leftmost = leftmostTMP[0]

            if rightmostTMP[0] > rightmost:
                rightmost = rightmostTMP[0]

            if topmostTMP[1] < topmost:
                topmost = topmostTMP[1]

            if bottommostTMP[1] > bottommost:
                bottommost = bottommostTMP[1]

        width = rightmost - leftmost
        height = bottommost - topmost
        return width, height


