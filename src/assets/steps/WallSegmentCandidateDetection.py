import numpy as np
import cv2
import math
import random

from ..algorithms.utils import distance, middle, mean_angle
from .Base import Base
class WallSegmentCandidateDetection(Base):
    def __init__(self, mask, Ma, row, col, debug=True):
        super().__init__(debug)
        self.mask = mask
        self.row = row
        self.col = col
        self.Ma = Ma
        self.Ls = []
        self.Lg = []
        self.H = np.zeros((self.row, self.col, 3), np.uint8)
        self.V = np.zeros((self.row, self.col, 3), np.uint8)
        self.D = np.zeros((self.row, self.col, 3), np.uint8)
        self.E = np.zeros((self.row, self.col, 3), np.uint8)

        self.B = np.zeros((self.row, self.col), np.uint8)

        self.directions = {
            'v': [(0, 1), (0, -1)],
            'h': [(1, 0), (-1, 0)],
            'd': [(1, -1), (-1, 1)],
            'e': [(1, 1), (-1, -1)],
        }

        self.test()

    def test(self):
        for rowEl in range(self.row):
            for colEl in range(self.col):
                element = self.Ma[rowEl][colEl]
                if element==-1 or self.mask[rowEl][colEl]==0:
                    continue
                elif 67.5<=math.fabs(element)<=112.5:
                    # вертикаль
                    self.setPixelInProjection(rowEl, colEl, 'v')
                elif -22.5<=element<=22.5 or math.fabs(element)>=157.5:
                    # горизонт
                    self.setPixelInProjection(rowEl, colEl, 'h')
                elif 22.5<element<67.5 or -157.5<element<-112.5:
                    # диагональ фронт
                    self.setPixelInProjection(rowEl, colEl, 'd')
                elif -67.5<element<-22.5 or 112.5<element<157.5:
                    # диагональ бэк
                    self.setPixelInProjection(rowEl, colEl, 'e')
                else:
                    print('Какой-то косяк i-{}, j-{}, element-{}'.format(rowEl, colEl, element))
        self.show('start_step5', self.mask)
        # self.show('vert', self.V)
        # self.show('horiz', self.H)
        # self.show('diagonal', self.D)
        # self.show('back', self.E)

        self.PixelProjection()

        self.show('vert_2', self.V)
        self.show('horiz_2', self.H)
        self.show('diagonal_2', self.D)
        self.show('back_2', self.E)

        self.lineExtraction()

    def setPixelInProjection(self, i, j, mode):
        if mode=='v':
            self.V[i][j] = [0, 0, 255]
        elif mode=='h':
            self.H[i][j] = [0, 255, 255]
        elif mode=='d':
            self.D[i][j] = [0, 255, 0]
        elif mode=='e':
            self.E[i][j] = [255, 0, 0]
        else:
            print('Какой-то косяк i-{}, j-{}, mode-{}'.format(i, j, mode))


    def PixelProjection(self):

        self.projectionDirection(self.V, 'v')
        self.projectionDirection(self.H, 'h')
        self.projectionDirection(self.D, 'd')
        self.projectionDirection(self.E, 'e')


    def projectionDirection(self, maskD, mode):
        row = self.row
        col = self.col
        for i in range(row):
            for j in range(col):
                element = maskD[i][j]
                if sum(element)>0:
                    self.projectionOneDirection(maskD, i, j, mode, 0)
                    self.projectionOneDirection(maskD, i, j, mode, 1)


    def projectionOneDirection(self, maskD, i, j, mode, to):
        row = self.row
        col = self.col
        direction = self.directions[mode]
        m = i
        n = j
        check = True
        while check:
            self.setPixelInProjection(m, n, mode)
            m = m + direction[to][1]
            n = n + direction[to][0]
            # not (0 <= m < row and 0 <= n < col and self.mask[m][n] > 0 and maskD[m][n] == 0):
            if self.isNotConNeigh(row, col, m, n, maskD):
                check = False

    def isNotConNeigh(self, row, col, m, n, mask):

        if not (0 <= m < row):
            return True

        if not (0 <= n < col):
            return True

        if not (self.mask[m][n] > 0):
            return True

        if not (sum(mask[m][n]) == 0):
            return True

        return False


    def lineExtraction(self):
        row = self.row
        col = self.col
        Ma = self.Ma
        mask = self.mask
        anglesH = []
        anglesV = []
        anglesD = []
        anglesE = []
        for i in range(row):
            for j in range(col):
                element = Ma[i][j]
                if element == -1:
                    continue
                self.B[i][j] = 1
                if sum(self.H[i][j]) > 0:
                    anglesH.append(element)
                if sum(self.V[i][j]) > 0:
                    anglesV.append(element)
                if sum(self.D[i][j]) > 0:
                    anglesD.append(element)
                if sum(self.E[i][j]) > 0:
                    anglesE.append(element)
        avgH = round(mean_angle(anglesH), 12)
        avgV = round(mean_angle(anglesV), 12)
        avgD = round(mean_angle(anglesD), 12)
        avgE = round(mean_angle(anglesE), 12)
        # self.print('avgH', avgH)
        # self.print('avgV', avgV)
        # self.print('avgD', avgD)
        # self.print('avgE', avgE)
        sliceTypeH = 'vertical' if 45<=avgH<135 else 'horizontal'
        sliceTypeV = 'vertical' if 45<=avgV<135 else 'horizontal'
        sliceTypeD = 'vertical' if 45<=avgD<135 else 'horizontal'
        sliceTypeE = 'vertical' if 45<=avgE<135 else 'horizontal'
        # self.print('sliceTypeH', sliceTypeH)
        # self.print('sliceTypeV', sliceTypeV)
        # self.print('sliceTypeD', sliceTypeD)
        # self.print('sliceTypeE', sliceTypeE)
        self.ScanSlice(self.H, sliceTypeH)
        self.ScanSlice(self.V, sliceTypeV)
        self.ScanSlice(self.D, sliceTypeD)
        self.ScanSlice(self.E, sliceTypeE)
        # self.print('Lg', self.Lg)
        self.testRes = np.zeros((self.row, self.col, 3), np.uint8)
        for i in range(len(self.Lg)):
            group = self.Lg[i]
            color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
            for slice in group:
                p0 = slice['p0']
                p1 = slice['p1']
                if p0[0] == p1[0]:
                    dif = 1
                    sim = 0
                else:
                    dif = 0
                    sim = 1
                p = [0,0]
                p[dif] = p0[dif]
                p[sim] = p0[sim]
                while True:
                    if p[dif]>=p1[dif]:
                        break
                    self.testRes[p[0]][p[1]] = color
                    p[dif] = p[dif]+1
        self.show('testRes', self.testRes)


    def ScanSlice(self, matrix, sliceType):
        directionsS = []
        directionsG = []
        if sliceType == 'horizontal':
            directionsS = self.directions['h']
            directionsG = self.directions['v']
        elif sliceType == 'vertical':
            directionsS = self.directions['v']
            directionsG = self.directions['h']
        self.SliceGroupDetection(matrix, directionsS, directionsG)

    def SliceGroupDetection(self, matrix, toS, toG):
        row = self.row
        col = self.col
        for i in range(row):
            for j in range(col):
                element = matrix[i][j]
                if sum(element) > 0 and self.B[i][j] == 1:
                    group = []
                    m = i
                    n = j
                    s0 = self.ScanHorAndVertSlice(matrix, m, n, toS)
                    if s0['length'] < 2:
                        continue


                    while True:
                        s = self.ScanHorAndVertSlice(matrix, m, n, toS, True)
                        last = None
                        if len(group)>0:
                            last = group[len(group)-1]
                        if not self.shouldContinue(group, s, last):
                            break
                        group.append(s)
                        m = round(s['center'][0]) + toG[1][0]
                        n = round(s['center'][1]) + toG[1][1]

                    s0 = group[0]
                    m = round(s0['center'][0]) + toG[0][0]
                    n = round(s0['center'][1]) + toG[0][1]
                    s0 = self.ScanHorAndVertSlice(matrix, m, n, toS)
                    if s0['length'] < 2:
                        continue
                    while True:
                        s = self.ScanHorAndVertSlice(matrix, m, n, toS, True)
                        first = None
                        if len(group) > 0:
                            first = group[0]
                        if not self.shouldContinue(group, s, first):
                            break
                        group.insert(0, s)
                        m = round(s['center'][0]) + toG[1][0]
                        n = round(s['center'][1]) + toG[1][1]


                    self.Lg.append(group)

    def ScanHorAndVertSlice(self, matrix, i, j, toS, change=False):
        if self.B[i][j] != 1:
            return {'length': 0, 'center': (0, 0), 'p0': (0, 0), 'p1': (0, 0)}
        m=i
        n=j
        slice = {
            'p0': (m, n),
            'p1': (m, n)
        }

        if change:
            self.B[m][n] = 2

        check = True
        while check:
            m = m + toS[1][0]
            n = n + toS[1][1]
            if self.B[m][n]==1 and sum(matrix[m][n]) > 0:
                slice['p0'] = (m, n)
                if change:
                    self.B[m][n] = 2
            else:
                check = False
        m = i
        n = j
        check = True
        while check:
            m = m + toS[0][0]
            n = n + toS[0][1]
            if self.B[m][n] == 1 and sum(matrix[m][n]) > 0:
                slice['p1'] = (m, n)
                if change:
                    self.B[m][n] = 2
            else:
                check = False
        slice['center'] = middle(slice['p0'], slice['p1'])
        slice['length'] = distance(slice['p0'], slice['p1'])
        return slice


    def shouldContinue(self, group, s, another):
        if s['length']<2:
            return False
        length = len(group)
        if another != None and length>0:
            if (another['length'])/2 > s['length']:
                return False

        return True




    # def ScanHorAndVertSlice(self, matrix, i, j, toS):
    #
    #     row = self.row
    #     col = self.col
    #     for i in range(row):
    #         for j in range(col):
    #             element = matrix[i][j]
    #             if sum(element) > 0 and self.B[i][j] != 2:
    #                 m=i
    #                 n=j
    #                 slice = {
    #                     'p0': (m, n),
    #                     'p1': (m, n)
    #                 }
    #                 self.B[m][n] = 2
    #                 check = True
    #                 while check:
    #                     m = m + toS[1][0]
    #                     n = n + toS[1][1]
    #                     if self.B[m][n]==1:
    #                         slice['p0'] = (m, n)
    #                         self.B[m][n] = 2
    #                     else:
    #                         check = False
    #
    #                 check = True
    #                 while check:
    #                     m = m + toS[0][0]
    #                     n = n + toS[0][1]
    #                     if self.B[m][n] == 1:
    #                         slice['p1'] = (m, n)
    #                         self.B[m][n] = 2
    #                     else:
    #                         check = False
    #                 slice['center'] = middle(slice['p0'], slice['p1'])
    #                 slice['length'] = distance(slice['p0'], slice['p1'])
    #                 self.Ls.append(slice)











