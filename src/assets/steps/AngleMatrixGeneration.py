import numpy as np
import cv2
import math
import matplotlib as plt
from skimage import measure
from ..algorithms.utils import Average
from .Base import Base
class AngleMatrixGeneration(Base):
    def __init__(self, debug=True):
        super().__init__(debug)
