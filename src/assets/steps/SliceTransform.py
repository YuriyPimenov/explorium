import numpy as np
import cv2
import math
import matplotlib as plt
from .Base import Base
from scipy.spatial import distance
from ..algorithms.utils import distance
class SliceTransform(Base):
    def __init__(self, binary, debug=True):
        super().__init__(debug)
        '''
        input binary image A(m*n)
        pixel a = A[i][j], 1<=i<=m, 1<=j<=n ( value 0 or 1 ) 
        '''
        self.A = binary
        self.show('Input', binary)
        self.test()

    def test(self):
        # print("Наш пиксель = {0}".format(self.A[206][142]))
        row, col = self.A.shape
        #Slice Matrix
        self.Ms = np.zeros((row, col, 4), np.uint8)

        '''
        slices 
            horizontal (0) - ph
            vertical (90) - pv
            forward diagonal (45) - pd
            backward diagonal (135) - pe
        '''
        '''
        Дискриптор пикселя определяется так 
        p{i,j} = Ts(a{i,j}) = [ ph, pv, pd, pe ] , p{i,j} ∈ N
        Ts(a{i,j}) это элемент SliceTransform, дискриптор пикселя картинки
        i,j=r,c
        '''

        for i in range(row):
            for j in range(col):
                if int(self.A[i][j])==1:
                    descriptor = []
                    descriptor.append( self.getPh(self.A, i, j, row, col) )
                    descriptor.append( self.getPv(self.A, i, j, row, col) )
                    descriptor.append( self.getPd(self.A, i, j, row, col) )
                    descriptor.append( self.getPe(self.A, i, j, row, col) )
                    self.Ms[i][j] = descriptor
                    # self.Ms[i][j][0] = self.getPh(self.A, i, j, row, col)
                    # self.Ms[i][j][1] = self.getPv(self.A, i, j, row, col)
                    # self.Ms[i][j][2] = self.getPd(self.A, i, j, row, col)
                    # self.Ms[i][j][3] = self.getPe(self.A, i, j, row, col)

        print(self.Ms)

    '''
    ph = 
    min{c | (c > j) ∧ (a{rc} = 0)} − 
    max{c | (a{rc} = 0) ∧ (c < j)} ,r = i
    '''
    def getPh(self, A=[], i=0, j=0, row=0, col=0):
        r = i

        rightSlice = c = j

        while True:
            c = c+1

            # Если мы вышли за предела массива в права
            if c > col-1:
                rightSlice = c
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a)==0:
                rightSlice = c
                break

        leftSlice = c = j
        while True:
            c = c-1

            # Если мы вышли за предела массива в слево
            if c < 0:
                leftSlice = c
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                leftSlice = c
                break

        slice = rightSlice - leftSlice - 1

        return slice

    '''
    pv =
    min{r | (a{rc} = 0) ∧ (r > i)} − 
    max{r | (a{rc} = 0) ∧ (r < i)} ,c = j
    '''
    def getPv(self, A=[], i=0, j=0, row=0, col=0):
        c = j

        downSlice = r = i

        while True:
            r = r+1

            # Если мы вышли за предела массива вниз
            if r > row-1:
                downSlice = r
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a)==0:
                downSlice = r
                break

        upSlice = r = i
        while True:
            r = r-1

            # Если мы вышли за предела массива в верх
            if r < 0:
                upSlice = r
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                upSlice = r
                break

        slice = downSlice - upSlice - 1

        return slice

    '''
    pe = 
    || 
        min{(r, c) | (a{rc} = 0) ∧ (r > i) ∧ (r − i = c − j)} −
        max{(r, c) | (a{rc} = 0) ∧ (r < i) ∧ (r − i = c − j)} 
    ||
    '''
    def getPe(self, A=[], i=0, j=0, row=0, col=0):

        bottomRight = [i, j]
        r = i
        c = j
        while True:
            r = r + 1
            c = c + 1
            # Если мы вышли за предела массива по диагонали вниз-право
            if r > row-1 or c > col-1:
                bottomRight = [r, c]
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                bottomRight = [r, c]
                break

        topLeft = [i, j]
        r = i
        c = j
        while True:
            r = r - 1
            c = c - 1

            # Если мы вышли за предела массива в верх-лево
            if r < 0 or c < 0:
                topLeft = [r, c]
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                topLeft = [r, c]
                break


        slice = math.floor( distance(bottomRight, topLeft) - 1)
        # v1 = (bottomRight[0], bottomRight[1])
        # v2 = (topLeft[0], topLeft[1])
        # dst = distance.euclidean((bottomRight[0], bottomRight[1]), (topLeft[0], topLeft[1]))
        # dst = np.linalg.norm(bottomRight - topLeft)
        # a = np.array(bottomRight)
        # b = np.array(topLeft)
        # dis = plt.mlab.dist(a, b)
        return slice

    '''
    pd = 
    ||
        min{(r, c) | ( a{rc} = 0) ∧ (r > i) ∧ (r − i = c − j)} −
        max{(r, c) | ( a{rc} = 0) ∧ (r < i) ∧ (r − i = c − j)}
    ||
    Тут якобы матрица повёрнута на 90 градусов
    '''
    def getPd(self, A=[], i=0, j=0, row=0, col=0):

        topRight = [i, j]
        r = i
        c = j
        while True:
            r = r - 1
            c = c + 1
            # Если мы вышли за предела массива по диагонали верх-право
            if r < 0 or c > col-1:
                topRight = [r, c]
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                topRight = [r, c]
                break

        bottomLeft = [i, j]
        r = i
        c = j
        while True:
            r = r + 1
            c = c - 1

            # Если мы вышли за предела массива в вниз-лево
            if r > row-1 or c < 0:
                bottomLeft = [r, c]
                break

            a = A[r][c]
            # Если мы наткнулись на пустой пиксель
            if int(a) == 0:
                bottomLeft = [r, c]
                break


        slice = math.floor( distance(topRight, bottomLeft) - 1)
        return slice