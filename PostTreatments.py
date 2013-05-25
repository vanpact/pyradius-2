#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: PostTreatments
        :platform: Unix, Windows
        :synopsis: module which contains the functions for the post treatments
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

import numpy

def computeAngle(angleAponeurosis, angleFiber):
	"""		
	Compute a pennation angle from an aponeurosis orientation and a muscle fascicle orientation
	
	:param angleAponeurosis: The aponeurosis orientation
	:type angleAponeurosis: float
	:param angleFiber: The muscle fiber orientation
	:type angleFiber: float
	:return: the pennation angle
	:rtype: float
	"""
	sina = numpy.sin(angleAponeurosis)*numpy.cos(angleFiber)-numpy.sin(angleFiber)*numpy.cos(angleAponeurosis)
	cosa = numpy.cos(angleAponeurosis)*numpy.cos(angleFiber)+numpy.sin(angleFiber)*numpy.sin(angleAponeurosis)
	return numpy.degrees(numpy.abs(numpy.arctan2(sina, cosa)))

def mergeMaxImages(imgList):
	"""		
	Compare the pixel at the same position in a list of image and create an image with the maximum value for each pixel
	
	:param imgList: The list of images
	:type imgList: list
	:return: the processed image
	:rtype: Numpy array
	"""
	finalImg = numpy.zeros_like(imgList[0])
	for img in imgList:
		finalImg = numpy.maximum(finalImg, img)
	return finalImg