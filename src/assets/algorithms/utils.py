import random
import math


def distance(a,b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_between(a,c,b):
    return distance(a,c) + distance(c,b) == distance(a,b)

def middle(a,b):
    cX = a[0] + (b[0] - a[0]) * 0.5
    cY = a[1] + (b[1] - a[1]) * 0.5
    return (cX, cY)

def getAngleFor3Points(pnt0, pnt1, pnt2):
    # векторы
    va = { "x":(pnt1[0] - pnt0[0]), "y":(pnt1[1] - pnt0[1])}
    vb = { "x":(pnt2[0] - pnt0[0]), "y":(pnt2[1] - pnt0[1])}
    #скалярное произведение векторов
    ab = va['x'] * vb['x'] + va['y'] * vb['y']
    #модули векторов
    mva = math.sqrt(va['x'] * va['x'] + va['y'] * va['y'])
    mvb = math.sqrt(vb['x'] * vb['x'] + vb['y'] * vb['y'])
    #косинус искомого угла
    #Это делаем на всякий случай, если вдруг произведение модулей векторов будет 0, то раз на 0 делить нельзя, будет ошибка
    pmv = 1
    if (int)(mva * mvb)==0:
        pmv = 1
    else:
        pmv = mva * mvb

    cosin = round(ab / pmv * 10000) * 0.0001
    angleRad = math.acos(cosin)

    angleDeg = round(angleRad * 180 / math.pi)

    return angleDeg

def Average(lst):
    return sum(lst) / len(lst)