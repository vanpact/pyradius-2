#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Oct 12, 2012

@author: yvesremi
"""

from PyQt4 import QtGui
import numpy, cv2
import gc, sys, debugsp

class ImageConverter(object):
    """Class to convert a numpy array to QImage and a QImage to a Numpy array."""


    def __init__(self):
        """Constructor"""
    
    @staticmethod
    def qimageToNdarray(img, colorToGray=False):
        """Convert a QImage to a Numpy array
        
        :param img: The QImage to be converted
        :type img: QImage
        :param colorToGray: if true, convert the image to grayScale
        :type colorToGray: bool
        :return: The converted image
        :rtype: Numpy array
        """
        if(isinstance(img, QtGui.QImage)):
            imageShape = (img.height(), img.width())
            temporaryShape = (img.height(), img.bytesPerLine() * 8 / img.depth())
            if img.format() in (QtGui.QImage.Format_ARGB32_Premultiplied, QtGui.QImage.Format_ARGB32, QtGui.QImage.Format_RGB32):
                imageShape += (4, )
                temporaryShape += (4, )
            elif (img.format()==QtGui.QImage.Format_RGB888):
                imageShape += (3, )
                temporaryShape += (3, )
            else:
                raise ValueError("Only 32 and 24 bits RGB and ARGB images are supported.")
            buf = img.bits().asstring(img.numBytes())
            ndimg = numpy.frombuffer(buf, numpy.uint8).reshape(temporaryShape)
            ndimg = ndimg[:, :, (2, 1, 0)]
            if imageShape != temporaryShape:
                ndimg = ndimg[:,:imageShape[1]]
            if img.format() == QtGui.QImage.Format_RGB32:
                ndimg = ndimg[...,:3]
            if(colorToGray):
                return ndimg[:, :, 0]*0.299+ndimg[:, :, 1]*0.587+ndimg[:, :, 2]*0.114
            return ndimg
        else:
            raise TypeError('Argument 1 must be a QtGui.QImage') 
            return 0
        
    @staticmethod
    def ndarrayToQimage(ndimg, form=QtGui.QImage.Format_RGB888):
        """Convert a Numpy array to a QImage

        :param ndimg: The Numpy array to be converted
        :type ndimg: Numpy array
        :param form: The QImage pixel format
        :type form: int
        :return: The converted image
        :rtype: QImage
        """
        if(isinstance(ndimg, numpy.ndarray)):
            ndimg1 = numpy.asarray(ndimg, numpy.uint8)
            if(len(ndimg1.shape)==2):#Grayscale images
                ndimg1 = numpy.dstack((ndimg1, numpy.copy(ndimg1), numpy.copy(ndimg1)))#cv2.cvtColor(ndimg1, cv2.cv.CV_GRAY2RGB)
#                ndimg = numpy.resize(ndimg,(ndimg.shape[0], ndimg.shape[1], 3))
            shape=ndimg1.shape
            ndimg3 = numpy.ravel(ndimg1)
            ndimg3.tostring()
            return QtGui.QImage(ndimg3.data, shape[1], shape[0], form)
        else:
            raise TypeError('Argument 1 must be a numpy.ndarray') 
            return None

#if __name__ == '__main__':
#    app = QtGui.QApplication(sys.argv)
#    image = QtGui.QImage('testGoodSize.png')
#    ndarray=ImageConverter.qimageToNdarray(image, True)
#    ImageConverter.ndarrayToQimage(ndarray).save('testConversion.png')