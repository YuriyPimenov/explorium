import cv2
from .steps.Preprocessing import Preprocessing
from .steps.SliceTransform import SliceTransform
from .steps.SliceThicknessFilter import SliceThicknessFilter
from .steps.AngleMatrixGeneration import AngleMatrixGeneration


class Explorium:
    def __init__(self, filepath):
        self.filepath = filepath

    def run(self):
        '''
        Шаг первый: первичная обработка
        '''
        self.preproc = Preprocessing(self.filepath, debug=False)
        '''
        Шаг второй: создание 8-битной 4 канальной матрицы
        '''
        self.sliceTrans = SliceTransform(self.preproc.result1, debug=False)
        '''
        Шаг третий: Обработка полученных дискрипторов
        '''
        self.sliceThickFilter = SliceThicknessFilter(self.sliceTrans.A, self.sliceTrans.Ms, self.preproc.row , self.preproc.col, self.preproc.mask, debug=False)
        '''
        Шаг четвертый: Создать матрицу, которая будет указывать какой поворот у стены содержащая текущий пиксель
        '''
        self.angleMatrixGen = AngleMatrixGeneration(debug=True)

        cv2.waitKey(0)
        cv2.destroyAllWindows()