import sys
import os
try:
    from .assets import Explorium
except (SystemError, ImportError):
    from assets import Explorium

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = 'images'

# FILE_NAME = '123.jpg'
FILE_NAME = 'plan.jpg'
# FILE_NAME = 'new.jpg'
# FILE_NAME = 'level0.jpg'
# FILE_NAME = 'level1.jpg'
# FILE_NAME = 'level2.jpg'
# FILE_NAME = 'level3.jpg'
# FILE_NAME = 'level4.jpg'
# FILE_NAME = 'level5.jpg'
# FILE_NAME = 'floor1.jpg'


filepath = os.path.join(BASE_DIR, 'src', DIR, FILE_NAME)
def main():
    exp = Explorium.Explorium(filepath)
    exp.run()


if __name__ == '__main__':
    sys.exit(main())