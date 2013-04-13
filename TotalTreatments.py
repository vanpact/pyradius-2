import Treatments
import PostTreatments
import numpy
import cv2

class LKMethod(object):
	def __init__(self, Aponeurosises, fiber):
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[1]
		
		self.treatmentsList = []
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(Treatments.MuscleTracker2(lines=Aponeurosises, fiber = fiber))
		
	def compute(self, img):
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
			
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)
		return(img, {'UpperPennationAngle':pennationAngleUp, 'LowerPennationAngle':pennationAngleLow})
			
class EllipseMethod(object):
	def __init__(self, Aponeurosises):
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[1]
		
		self.treatmentsList = []
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(Treatments.blobDetectionTreatment(lines=Aponeurosises))
		
	def compute(self, img):
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
			
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)
		return(img, {'UpperPennationAngle':pennationAngleUp, 'LowerPennationAngle':pennationAngleLow})

class RadonMethod(object):
	def __init__(self, Aponeurosises, manySamples=False):
		upperAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[0]
		lowerAponeurosis = numpy.argsort(numpy.asarray([Aponeurosises[0][0].y(), Aponeurosises[1][0].y()]))[1]
		
		self.treatmentsList = []
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[upperAponeurosis]))
		self.treatmentsList.append(Treatments.AponeurosisTracker(line=Aponeurosises[lowerAponeurosis]))
		self.treatmentsList.append(Treatments.testRadon(lines=Aponeurosises, manyCircles=manySamples))
		
	def compute(self, img):
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
			
		imgToMix=[numpy.copy(img) for _ in range(len(self.treatmentsList))]
		angle = numpy.zeros(len(self.treatmentsList), numpy.float)
		for i, treatment in enumerate(self.treatmentsList):
			imgToMix[i], angle[i] = treatment.compute(imgToMix[i])
		
		pennationAngleUp = PostTreatments.computeAngle(angle[0], angle[2])
		pennationAngleLow = PostTreatments.computeAngle(angle[1], angle[2])
		img = PostTreatments.mergeMaxImages(imgToMix)
		return(img, {'UpperPennationAngle':pennationAngleUp, 'Lower pennationAngle':pennationAngleLow})

class junctionComputation(object):
	def __init__(self, limits, AponeurosisThickness=45):
		self.treatment = Treatments.seamCarving(limits, AponeurosisThickness)
		
	def compute(self, img):
		if(len(img.shape)>2):
			img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
		img, center = self.treatment.compute(img)
		return(img, {'center':center})