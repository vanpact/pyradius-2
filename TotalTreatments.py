import Treatments
import PreTreatments
import PostTreatments

class pennationComputation(object):
	def __init__(self):
		self.treatment = []
		self.treatment.append(Treatments.AponeurosisTracker)
		self.treatment.append(Treatments.AponeurosisTracker)
	def chooseEllipsoid(self, lines):
		self.treatment.append(Treatments.blobDetectionTreatment(lines))


class junctionComputation(object):