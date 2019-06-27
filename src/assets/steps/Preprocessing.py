import numpy as np
import cv2
from .Base import Base

import imutils
from skimage import measure
from pythonRLSA import rlsa
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

from ..algorithms.text_detection import runTD
from ..algorithms.utils import Average

class Preprocessing(Base):
    def __init__(self, filepath, debug=True):
        super().__init__(debug=debug)

        # Путь к картинке
        self.filepath = None
        # Оригинал
        self.image = None
        # Отображение стен на картинке
        self.mask_image = None
        # Маска со стенами
        self.mask = None
        # Результат первого этапа ( матрица со значениями 0 или 1 )
        self.result1 = None

        self.test6(filepath)


    def removeText(self, TD, mask):
        hsv = cv2.cvtColor(TD, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([60, 255, 255])
        upper_blue = np.array([60, 255, 255])
        maskTextDetect = cv2.inRange(hsv, lower_blue, upper_blue)
        maskTextDetect = cv2.bitwise_not(maskTextDetect)
        bit_xor = cv2.bitwise_and(mask, maskTextDetect)
        return bit_xor

    def rotate(self, img, center, angle90, scale, h, w):
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        return cv2.warpAffine(img, M, (h, w))

    def test2(self, filepath):
        self.filepath = filepath
        # Читаем картинку
        image = cv2.imread(filepath)  # reading the image
        self.image = image

        # Преобразуем в серый
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Превращаем в черно-белую картинку
        (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
        self.show('binary', binary)

        # Создаём пустой бланк, того же размера что и оригинал
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        # Ищем контуры ( так как изо-ние бинарное то ставим тильду )
        contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Получаем высоту каждого контура
        heights = [cv2.boundingRect(contour)[3] for contour in contours]
        # Получаем среднюю высоту
        avgheight = sum(heights) / len(heights)

        # Рисуем контуры у которых высота больше 5% высоты картинки
        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            # if h > 2 * avgheight:
            if h > 0.05 * image.shape[0]:
                cv2.drawContours(mask, [c], -1, 0, 1)
        self.show('filter', mask)


        # Выполняем горизонтальную и вертикальную эвристику
        # value = max(math.ceil(x / 100), math.ceil(y / 100)) + 25  # heuristic
        # mask = rlsa.rlsa(mask, True, True, value)  # rlsa application
        # x, y = mask.shape
        # value = math.ceil(x / 100) + 20  # heuristic
        # mask = rlsa.rlsa(mask, True, False, value)
        # value = math.ceil(y / 100) + 20
        # mask = rlsa.rlsa(mask, False, True, value)


        # На оригинал накладываем маску
        mask = cv2.bitwise_not(mask)
        self.show('rlsah', mask)
        mask_image = cv2.bitwise_and(image, image, mask=mask)
        self.show('mask_image', mask_image)

        # Может пригодится
        # gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        # (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
        # cv2.imshow('binary1111', binary)

        # Делаем картинку со значениями 0 и 1
        result1 = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.show('result1', result1)

        self.mask_image = mask_image
        self.mask = mask
        self.result1 = result1
        # contours2, hierarchy2 = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # mask2 = np.ones(image.shape, dtype="uint8") * 255  # blank 3 layer image
        # for contour2 in contours2:
        #     [x, y, w, h] = cv2.boundingRect(contour2)
        #     # if w > 0.20 * image.shape[1]:  # width heuristic applied
        #     title = image[y: y + h, x: x + w]
        #     mask2[y: y + h, x: x + w] = title  # copied title contour onto the blank image
        #     image[y: y + h, x: x + w] = 255  # nullified the title contour on original image
        #
        # cv2.imshow('title', mask2)




        # for contour in contours:
        #     """
        #     draw a rectangle around those contours on main image
        #     """
        #     [x, y, w, h] = cv2.boundingRect(contour)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)



    def test1(self,filepath):
        self.filepath = filepath

        origin = cv2.imread(filepath)
        img = np.copy(origin)
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)

        # text detection
        img_text_detect = runTD(img_blur)
        mask_without_text = self.removeText(img_text_detect, mask)
        cv2.imshow("bitwise_or1", mask_without_text)

        img_without_text = cv2.cvtColor(mask_without_text, cv2.COLOR_GRAY2RGB)
        img_text_detect2 = runTD(img_without_text)
        mask_without_text2 = self.removeText(img_text_detect2, mask_without_text)
        cv2.imshow("bitwise_or2", mask_without_text2)

        img_without_text2 = cv2.cvtColor(mask_without_text2, cv2.COLOR_GRAY2RGB)
        (h, w) = img.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)
        img_without_text2 = self.rotate(img_without_text2, center, 270, 1.0, h, w)

        img_text_detect3 = runTD(img_without_text2)
        cv2.imshow("img_text_detect3", img_text_detect3)
        cv2.imshow("mask_without_text2", mask_without_text2)
        # mask_without_text3 = self.removeText(img_text_detect3, mask_without_text2)
        # cv2.imshow("bitwise_or3", mask_without_text3)

        # img_without_text2 = cv2.cvtColor(mask_without_text2, cv2.COLOR_GRAY2RGB)
        # img_text_detect3 = runTD(img_without_text2)
        # mask_without_text3 = self.removeText(img_text_detect3, mask_without_text2)
        # cv2.imshow("bitwise_or3", mask_without_text3)

        # Возможно это оставим
        # norm_image = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # print(norm_image)
        # cv2.imshow("Image", gray)
        # cv2.imshow("gray", gray)
        # cv2.imshow("mask", mask)
        # cv2.imshow("hsv", hsv)
        # cv2.imshow("norm_image", norm_image)
        # cv2.imwrite('./res.jpg', mask)
        # gray = cv2.cvtColor(norm_image, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("mask", mask)
        # maskTextDetect = cv2.bitwise_not(maskTextDetect)
        # cv2.imshow("maskTextDetect", maskTextDetect)
        # bit_xor = cv2.bitwise_and(mask, maskTextDetect)
        # cv2.imshow("bitwise_or", bit_xor)
        # cv2.imshow("binary", binary)

    def test3(self, filepath):
        def nothing(x):
            pass

        canny = 95
        morph = 8
        CANNY = canny
        MORPH = morph

        self.filepath = filepath
        # Читаем картинку
        origin_image = cv2.imread(filepath)  # reading the image
        self.image = origin_image


        img_gray = cv2.GaussianBlur(origin_image, (3, 3), cv2.BORDER_REPLICATE)

        edges = cv2.Canny(img_gray, 10, 128)
        self.show('edges', edges)



        cv2.namedWindow("Image")
        cv2.createTrackbar("morphSlider", "Image", 2, 128, nothing)
        cv2.createTrackbar("dilationSlider", "Image", 1, 30, nothing)
        cv2.createTrackbar("erosionSlider", "Image", 1, 30, nothing)
        cv2.createTrackbar("kernelSlider", "Image", 1, 10, nothing)

        while True:
            valueMor = cv2.getTrackbarPos("morphSlider", "Image")
            valueDilation = cv2.getTrackbarPos("dilationSlider", "Image")
            valueErosion = cv2.getTrackbarPos("erosionSlider", "Image")
            valueKernel = cv2.getTrackbarPos("kernelSlider", "Image")

            kernel = np.ones((valueKernel, valueKernel), np.uint8)
            dilation = cv2.dilate(edges, kernel, iterations=valueDilation)
            erosion = cv2.erode(edges, kernel, iterations=valueErosion)
            self.show('dilation', dilation)
            self.show('erosion', erosion)


            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (valueMor, valueMor))

            closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
            self.show('closed', closed)

            key = cv2.waitKey(100)
            if key == 27:
                break



        # (3, 3), 1 - 9хор 1ср 1пло

        #
        #

        # _width = origin_image.shape[1] * 0.7
        # _height = origin_image.shape[0] * 0.7
        # _margin = 0.0
        #
        # corners = np.array(
        #     [
        #         [[_margin, _margin]],
        #         [[_margin, _height + _margin]],
        #         [[_width + _margin, _height + _margin]],
        #         [[_width + _margin, _margin]],
        #     ]
        # )

        # pts_dst = np.array(corners, np.float32)


        # contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # img_croped = None
        #
        # for cont in contours:
        #
        #     if cv2.contourArea(cont) > 5000:
        #
        #         arc_len = cv2.arcLength(cont, True)
        #
        #         approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)
        #
        #         if (len(approx) == 4):
        #             pts_src = np.array(approx, np.float32)
        #
        #             h, status = cv2.findHomography(pts_src, pts_dst)
        #             out = cv2.warpPerspective(origin_image, h,
        #                                       (int(_width + _margin * 2), int(_height + _margin * 2)))
        #             cv2.drawContours(origin_image, [approx], -1, (0, 255, 0), 4)
        #             k = cv2.isContourConvex(cont)
        #             # обрезаем изображение
        #             x1_y1 = np.min(approx, axis=0)
        #             x2_y2 = np.max(approx, axis=0)
        #             x1 = x1_y1[0, 0]
        #             y1 = x1_y1[0, 1]
        #             x2 = x2_y2[0, 0]
        #             y2 = x2_y2[0, 1]
        #             img_croped = origin_image[y1:y2, x1:x2]
        #
        #         else:
        #             pass

    def test4(self, filepath):
        self.filepath = filepath
        # Читаем картинку
        origin_image = cv2.imread(filepath)  # reading the image
        self.image = origin_image

        binary_img = cv2.GaussianBlur(origin_image, (3, 3), cv2.BORDER_REPLICATE)

        edges = cv2.Canny(binary_img, 50, 120)
        cv2.imshow('image', edges)
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     cv2.destroyAllWindows()
        lines_data = cv2.HoughLines(edges, 1, np.pi / 180, 110)

        parallel_lines = []
        vertical_lines = []
        for rho, theta in lines_data[0]:
            # print 'rho:  '+str(rho)+'theta:  '+str(theta)
            if 2 > theta > 1:
                vertical_lines.append([theta, rho])
            elif theta < 1:
                parallel_lines.append([theta, rho])
            elif theta > 3:
                parallel_lines.append([theta, rho])

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('image', edges)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()

        # vertical_lines = sorted(vertical_lines, key=lambda x: abs(x[1]))
        # parallel_lines = sorted(parallel_lines, key=lambda x: abs(x[1]))
        # return vertical_lines, parallel_lines

    def removeNoice(self, edges, k=1):
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        areas = [cv2.contourArea(contour) for contour in contours]
        # Получаем высоту каждого контура
        heights = [cv2.boundingRect(contour)[3] for contour in contours]
        # Получаем среднюю высоту
        avgheight = sum(heights) / len(heights)
        avgArea = sum(areas) / len(areas)
        # Рисуем контуры у которых высота больше 5% высоты картинки
        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            # area = cv2.contourArea(c)
            # if area < avgArea/3:
            if h < avgheight*k:
                # [x, y, w, h] = cv2.boundingRect(c)
                edges[y: y + h, x: x + w] = 0  # nullified the title contour on original image
        return edges

    def test5(self, filepath):

        self.filepath = filepath
        # Читаем картинку
        origin_image = cv2.imread(filepath)  # reading the image
        self.image = origin_image
        self.show('origin_image', origin_image)

        img_gray = cv2.GaussianBlur(origin_image, (3, 3), cv2.BORDER_REPLICATE)

        edges = cv2.Canny(img_gray, 10, 128)
        self.show('edges', edges)

        edges = self.removeNoice(edges, 1)
        self.show('edges2', edges)

        # edges = self.removeNoice(edges, 1/7)
        # self.show('edges3', edges)



        kernel = np.ones((1, 1), np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=1)
        erosion = cv2.erode(edges, kernel, iterations=1)
        self.show('dilation', dilation)
        self.show('erosion', erosion)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        self.show('closed', closed)

        result1 = cv2.normalize(closed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.show('result1', result1)


        self.mask = closed
        self.result1 = result1

    def test6(self, filepath):

        self.filepath = filepath
        # Читаем картинку
        origin_image = cv2.imread(filepath)  # reading the image
        (h, w) = origin_image.shape[:2]
        self.row = h
        self.col = w
        self.image = origin_image
        self.show('origin_image', origin_image)

        # Преобразуем в серый
        gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

        # Превращаем в черно-белую картинку
        (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
        self.show('binary', binary)

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)

        dilation = cv2.dilate(erosion, kernel, iterations=5)


        dilation_not = cv2.bitwise_not(dilation)
        binary_not = cv2.bitwise_not(binary)

        res_sub = cv2.bitwise_or(binary, dilation_not)

        res_sub_not = cv2.bitwise_not(res_sub)


        #Удаляем маленькие кусочки и текст
        mask, dataNames, dataValues = self.removeLittlePieces(res_sub_not)

        # for i in range(len(dataNames)):
        #     if dataValues[i] > int(avgCountPixels)*2:
        #         mask = cv2.add(mask, masksLabel[i])
        #         self.show(str(i), masksLabel[i])

        self.show('mask', mask)
        mask_not = cv2.bitwise_not(mask)
        self.show('mask_not', mask_not)

        self.showHist(dataNames, dataValues, False)

        # Делаем картинку со значениями 0 и 1
        result1 = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.mask = mask
        self.result1 = result1

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

# bit_and = cv2.bitwise_and(img2, img1)
# bit_or = cv2.bitwise_or(img2, img1)
# bit_xor = cv2.bitwise_xor(img1, img2)
# bit_not = cv2.bitwise_not(img1)
# bit_not2 = cv2.bitwise_not(img2)

