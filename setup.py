from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys, os, py2exe
from glob import glob

def runSetup(setup_args):
    setup(
    name = 'Pyradius',
    version = '1.0',
    description = 'Angle Extractor for ultrasound images',
    author = 'Yves-Remi Van Eycke',
    author_email = 'yrvaneycke@gmail.com',
    url = 'https://code.google.com/p/pyradius/',
#     console=['main.py'], 
    **setup_args
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = ext_modules
    )

if __name__=='__main__':
    try:
        sys.stdout = open("nul", "w")
        sys.stderr = open("nul", "w")
        os.rename('Applier.py', 'Applier.pyx')
        os.rename('MainTreatments.py', 'MainTreatments.pyx')
        os.rename('PreTreatments.py', 'PreTreatments.pyx')
        os.rename('TotalTreatments.py', 'TotalTreatments.pyx')
        os.rename('imageConverter.py', 'imageConverter.pyx')
        os.rename('PostTreatments.py', 'PostTreatments.pyx')
        setup_args = {}
        setup_args['cmdclass'] = {'build_ext': build_ext}
        setup_args['ext_modules'] = [Extension("Applier", ["Applier.pyx"]),
                                         Extension("MainTreatments", ["MainTreatments.pyx"]), 
                                         Extension("PreTreatments", ["PreTreatments.pyx"]), 
                                         Extension("TotalTreatments", ["TotalTreatments.pyx"]),
                                         Extension("imageConverter", ["imageConverter.pyx"]),
                                         Extension("PostTreatments", ["PostTreatments.pyx"])]
        setup_args['data_files'] = [("data", glob(r'C:\Python27\lib\site-packages\scipy\integrate\_quadpack.pyd')), 
                  ("data", glob(r'C:\Python27\lib\site-packages\scipy\optimize\minpack2.pyd')), 
                  ("data", glob(r'C:\Python27\lib\site-packages\scipy\interpolate\dfitpack.pyd')), 
                  ("data", glob(r'C:\Python27\lib\site-packages\numpy-1.7.1-py2.7-win32.egg\numpy\fft\fftpack_lite.pyd')),
                  ("data", glob(r'C:\Python27\lib\site-packages\skimage\_shared\geometry.pyd')),
                  ("data", glob(r'C:\Program Files (x86)\CMake 2.8\bin\MSVCP90.dll')),
                  ("imageformats", glob(r'C:\Python27\Lib\site-packages\PyQt4\plugins\imageformats\qgif4.dll')),
                  ("Images", glob(r'Images\*.gif')),
                  ("Images", glob(r'Images\*.png')),
                  ("Images", glob(r'Images\*.jpg')),
                  ("Images", glob(r'Images\*.ico')),
                  ("", glob(r'gpl-3.0-standalone.html'))]
        runSetup(setup_args)
        setup_args2 = {}
        setup_args2['data_files'] = setup_args['data_files'].append([
                    ("", glob(r'build\lib.win32-2.7\Applier.pyd')),
                    ("", glob(r'build\lib.win32-2.7\MainTreatments.pyd')),
                    ("", glob(r'build\lib.win32-2.7\Treatments.pyd')),
                   ("", glob(r'build\lib.win32-2.7\PreTreatments.pyd')),
                   ("", glob(r'build\lib.win32-2.7\TotalTreatments.pyd')),
                   ("", glob(r'build\lib.win32-2.7\imageConverter.pyd')),
                   ("", glob(r'build\lib.win32-2.7\PostTreatments.pyd'))])
        setup_args2['options']={"py2exe": {"includes": ["sip", "skimage._shared.*", "scipy.sparse.csgraph._validation", "six", "openpyxl.workbook"]}}
        setup_args2['windows']=[{"script": "main.py"}]
        runSetup(setup_args2)
    finally:
        os.rename('Applier.pyx', 'Applier.py')
        os.rename('MainTreatments.pyx', 'MainTreatments.py')
        os.rename('PreTreatments.pyx', 'PreTreatments.py')
        os.rename('TotalTreatments.pyx', 'TotalTreatments.py')
        os.rename('imageConverter.pyx', 'imageConverter.py')
        os.rename('PostTreatments.pyx', 'PostTreatments.py')
        try:
            os.remove('dist/pyradius.exe')
        except:
            pass
        os.rename('dist/main.exe', 'dist/pyradius.exe')