import numpy as np
import cv2
import math

import matplotlib as plt
from skimage import measure
from ..algorithms.utils import Average, compass_to_rgb, mean_angle
from .Base import Base
class AngleMatrixGeneration(Base):
    def __init__(self, mask, Ms, row , col, debug=True):
        super().__init__(debug)
        self.mask = mask
        # self.mask = mask
        #Порогове значение для определение похожести
        self.threshSim = 10
        self.Ms = Ms
        self.row = row
        self.col = col
        # self.Ma = -1
        self.Ma = -1
        self.test()

    def test(self):
        self.show('start_step4', self.mask)
        self.testImg = np.zeros((self.row, self.col, 3), np.uint8)

        self.Ma = self.AngleMatrix(self.mask, self.Ms, self.row, self.col)

        self.show('visible angle', self.testImg)

        self.ConditionalBlur(self.mask, self.Ma, self.row, self.col)
        self.show('visible angle2', self.testImg)


    def AngleMatrix(self, mask, Ms, row, col):
        Ma = np.zeros((row, col), np.int)
        for rowEl in range(row):
            for colEl in range(col):
                element = Ms[rowEl][colEl]
                ph, pv, pd, pe = element
                if ph != 0 and pv != 0 and pd != 0 and pe != 0 and mask[rowEl][colEl]!=0:
                    if pd==pe and pd==ph or pd==pe and pv>ph:
                        Ma[rowEl][colEl] = 90
                    elif pd==pe and pd==pv or pd==pe and ph>pv:
                        Ma[rowEl][colEl] = 0
                    elif ph == pv and pe<0.1*pd:
                        Ma[rowEl][colEl] = 45
                    elif ph == pv and pd<0.1*pe:
                        Ma[rowEl][colEl] = 135
                    else:
                        if pd>pe:
                            L0 = (math.tan(pv/ph))**-1
                        else:
                            L0 = -1 * (math.tan(pv/ph))**-1

                        if ph>pv:
                            if pd>pe:
                                L1 = (math.tan(pd/pe))**-1 - math.pi/4
                            else:
                                L1 = (math.tan(pd/pe))**-1 + math.pi*3/4
                        else:
                            L1 = (math.tan(pe/pd))**-1 + math.pi/4

                        if L0<0:
                            L0 = L0 + math.pi
                        if L1<0:
                            L1 = L1 + math.pi

                        if 0.5 * math.fabs(ph-pv)>math.fabs(pd-pe):
                            Ma[rowEl][colEl] = math.degrees(L0)
                        elif 0.5 * math.fabs(pd-pe)>math.fabs(ph-pv):
                            Ma[rowEl][colEl] = math.degrees(L1)
                        else:
                            Ma[rowEl][colEl] = math.degrees(self.avgAngles([L1, L0]))
                else:
                    Ma[rowEl][colEl] = -1
                if Ma[rowEl][colEl]!=-1:
                    angle = 0
                    if Ma[rowEl][colEl]>=180:
                        angle = Ma[rowEl][colEl] - 180
                    else:
                        angle = Ma[rowEl][colEl] + 180
                    r, g, b = compass_to_rgb(angle)
                    self.testImg[rowEl][colEl] = [b, g, r]
                else:
                    self.testImg[rowEl][colEl] = [0, 0, 0]


        return Ma

    def ConditionalBlur(self, mask, Ma, row, col):
        for rowEl in range(row):
            for colEl in range(col):
                element = Ma.item((rowEl, colEl))
                if element!=-1:
                    neighbors = []
                    neighbors.append(Ma.item((rowEl-1, colEl-1)))
                    neighbors.append(Ma.item((rowEl, colEl-1)))
                    neighbors.append(Ma.item((rowEl+1, colEl-1)))
                    neighbors.append(Ma.item((rowEl-1, colEl)))
                    neighbors.append(Ma.item((rowEl+1, colEl)))
                    neighbors.append(Ma.item((rowEl-1, colEl+1)))
                    neighbors.append(Ma.item((rowEl, colEl+1)))
                    neighbors.append(Ma.item((rowEl+1, colEl+1)))
                    # neighbors.append(Ma[rowEl-1][colEl-1])
                    # neighbors.append(Ma[rowEl][colEl-1])
                    # neighbors.append(Ma[rowEl+1][colEl-1])
                    # neighbors.append(Ma[rowEl-1][colEl])
                    # neighbors.append(Ma[rowEl+1][colEl])
                    # neighbors.append(Ma[rowEl-1][colEl+1])
                    # neighbors.append(Ma[rowEl][colEl+1])
                    # neighbors.append(Ma[rowEl+1][colEl+1])

                    similars = list()
                    differents = list()

                    for i in range(8):
                        neighbor = neighbors[i]
                        if neighbor!=-1:
                            if self.isSimilar(neighbor, element):
                                similars.append(neighbor)
                            else:
                                differents.append(neighbor)
                    lenSimil = len(similars)
                    lenDiffer = len(differents)
                    if lenSimil>lenDiffer:
                        val = Ma.item((rowEl, colEl))
                        similars.append(val)
                        element = self.avgAngles(similars)
                    elif lenSimil<lenDiffer:
                        element = self.avgAngles(differents)

                    Ma[rowEl][colEl] = element

                    angle = 0
                    if element >= 180:
                        angle = element - 180
                    else:
                        angle = element + 180
                    r, g, b = compass_to_rgb(angle)
                    self.testImg[rowEl][colEl] = [b, g, r]

    def isSimilar(self, neighbor, element):
        dif = self.getDif(neighbor, element)
        return dif<self.threshSim

    def getDif(self, angle1, angle2):
        el1 = math.fabs(angle1 - angle2)
        el2 = math.fabs(angle1 - (angle2 + 180))
        dif = min([el1, el2])
        return dif

    def avgAngle(self, a1, a2):
        avg = 0
        if self.getDif(a1, a2)<=self.getDif(a1, a2+180):
            avg = (a1 + a2)/2
        else:
            avg = (a1 + a2 + 180) / 2
        return avg

    def avgAngles(self, angles):
        return round(mean_angle(angles), 12)


