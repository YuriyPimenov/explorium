import cv2
from .steps.Preprocessing import Preprocessing
from .Algorithms import Algorithms

class Explorium:
    def __init__(self, filepath):
        self.filepath = filepath

    def run(self):
        '''
        Шаг первый: первичная обработка
        '''
        self.preproc = Preprocessing(self.filepath)

        cv2.waitKey(0)
        cv2.destroyAllWindows()