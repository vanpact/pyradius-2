import numpy

def computeAngle(anglesAponeurosis, angleFiber):
	angleAp = numpy.mean(anglesAponeurosis)
	sina = numpy.sin(angleAp)*numpy.cos(angleFiber)-numpy.sin(angleFiber)*numpy.cos(angleAp)
	cosa = numpy.cos(angleAp)*numpy.cos(angleFiber)+numpy.sin(angleFiber)*numpy.sin(angleAp)
	return numpy.degrees(numpy.abs(numpy.arctan2(sina, cosa)))