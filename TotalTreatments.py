#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Oct 12, 2012

@author: yvesremi
"""

import MainTreatments
import PostTreatments
import numpy
import cv2
import time

class LKMethod(object):
	"""Class performing the pretreatments, treatments and post-treatments for the method using the Lucas-Kanade algorithm"""
	def __init__(self, Aponeurosises, fiber):
		"""
		Constructor
		
		:param Aponeurosises: The position of the extremities of the aponeurosises
		:type Aponeurosises: list of 2 tuple containing 2 QPoints each
		:param fiber: The position of the extremities of a muscle fascicle
		:type fiber: tuple containing 2 QPoints
		"""
		left0 = numpy.argsort(numpy.asarray([Aponeurosises[0][0].x(), Aponeurosises[0][1].x()]))[0]
		left1 = numpy.argsort(numpy.asarray([Aponeurosises[1][0].x(), Aponeurosises[1][1].x()]))[0]
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[1]
		self.treatmentsList = []
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(MainTreatments.MuscleTracker2(lines=Aponeurosises, fiber = fiber))
		
	def compute(self, img):
		"""
		Process one image
		
		:param img: The image to process
		:type img: Numpy array	
		:return: The processed image and a dictionnary containing the 2 pennation angles extracted (the key is the channel name)
		:rtype: Tuple
		"""
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
			
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)
		
		return(img, {'26 UpperPennationAngle':pennationAngleUp, '25 LowerPennationAngle':pennationAngleLow})
			
class EllipseMethod(object):
	"""Class performing the pretreatments, treatments and post-treatments for the method using the Ellipsoids to detect the muscle fascicle orientation"""
	def __init__(self, Aponeurosises):
		"""
		Constructor
		
		:param Aponeurosises: The position of the extremities of the aponeurosises
		:type Aponeurosises: list of 2 tuple containing 2 QPoints each
		"""
		left0 = numpy.argsort(numpy.asarray([Aponeurosises[0][0].x(), Aponeurosises[0][1].x()]))[0]
		left1 = numpy.argsort(numpy.asarray([Aponeurosises[1][0].x(), Aponeurosises[1][1].x()]))[0]
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[1]
		
		
		self.treatmentsList = []
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(MainTreatments.blobDetectionTreatment(lines=Aponeurosises))
		
	def compute(self, img):
		"""	
		Process one image
			
		:param img: The image to process
		:type img: Numpy array	
		:return: The processed image and a dictionnary containing the 2 pennation angles extracted(the key is the channel name)
		:rtype: Tuple
		"""
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
		
		
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)

		return(img, {'26 UpperPennationAngle':pennationAngleUp, '25 LowerPennationAngle':pennationAngleLow})

class RadonMethod(object):
	"""Class performing the pretreatments, treatments and post-treatments for the method using the Radon transform"""
	def __init__(self, Aponeurosises, manySamples=False):
		"""
		Constructor
		
		:param Aponeurosises: The position of the extremities of the aponeurosises
		:type Aponeurosises: list of 2 tuple containing 2 QPoints each
		:param manySamples: if True, use 3 samples instead of one but is three times slower
		:type manySamples: bool
		"""
		left0 = numpy.argsort(numpy.asarray([Aponeurosises[0][0].x(), Aponeurosises[0][1].x()]))[0]
		left1 = numpy.argsort(numpy.asarray([Aponeurosises[1][0].x(), Aponeurosises[1][1].x()]))[0]
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][left0].y(), Aponeurosises[1][left1].y()]))[1]
		self.treatmentsList = []
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(MainTreatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(MainTreatments.testRadon(lines=Aponeurosises, manyCircles=manySamples))
		
	def compute(self, img):
		"""		
		Process one image
				
		:param img: The image to process
		:type img: Numpy array	
		:return: The processed image and a dictionnary containing the 2 pennation angles extracted (the key is the channel name)
		:rtype: Tuple
		"""
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
			
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)
		
		return(img, {'26 UpperPennationAngle':pennationAngleUp, '25 LowerPennationAngle':pennationAngleLow})

class junctionComputation(object):
	"""Class performing the pretreatments, treatments and post-treatments in order to extract the junction of an aponeurosis"""
	def __init__(self, limits, firstApproximation):
		"""		
		Constructor
		
		:param limits: 2 points delimitating a rectangular region of interest
		:type limits: Tuple of 2 QPoints
		:param firstApproximation: First approximation of the center
		:type firstApproximation: QPoint
		:return: The processed image and a dictionnary containing the center of the junction extracted (the key is the channel name)
		:rtype: Tuple
		"""
		self.treatment = MainTreatments.seamCarving(limits, firstApproximation)
		
	def compute(self, img):
		"""		
		Process one image
		
		:param img: The image to process
		:type img: Numpy array	
		:return: The processed image and a dictionnary containing the center of the junction extracted (the key is the channel name)
		:rtype: Tuple
		"""
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
		img, center = self.treatment.compute(img)
		return(img, {'center':center})