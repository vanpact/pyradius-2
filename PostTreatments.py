import numpy

def computeAngle(angleAponeurosis, angleFiber):
	sina = numpy.sin(angleAponeurosis)*numpy.cos(angleFiber)-numpy.sin(angleFiber)*numpy.cos(angleAponeurosis)
	cosa = numpy.cos(angleAponeurosis)*numpy.cos(angleFiber)+numpy.sin(angleFiber)*numpy.sin(angleAponeurosis)
	return numpy.degrees(numpy.abs(numpy.arctan2(sina, cosa)))

def mergeMaxImages(imgList):
	finalImg = numpy.zeros_like(imgList[0])
	for img in imgList:
		finalImg = numpy.maximum(finalImg, img)
	return finalImg