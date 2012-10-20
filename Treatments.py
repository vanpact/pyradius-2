'''
Created on Oct 20, 2012

@author: yvesremi
'''
import cv2, numpy


class AbstractTreatment(object):
    def __init__(self):
        '''
        Constructor
        '''
        if(type(self) is AbstractTreatment):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(AbstractTreatment, self).__init__()
        
    def compute(self, img):
            raise NotImplementedError( "The method need to be implemented" )
        
class blobDetectionTreatment(AbstractTreatment):
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        
    def extractData(self, contours):
        #find angle
        alpha = []
        for c in contours:
            xy = c[:,0,:]
            (vx, vy, x0, y0) = cv2.fitLine(points=xy, distType=cv2.cv.CV_DIST_L2, param=0, reps=10, aeps=0.01)
            alpha.append(numpy.arctan2(vy,vx))
        angles = numpy.asarray(alpha)
    
        #grouping all segments
        data = numpy.zeros((0,3))
        for alpha,c in zip(angles,contours):
            x = c[:,0,0]
            y = c[:,0,1]
            if alpha<1 and alpha>-1:
                a = numpy.ones_like(x)*alpha
                data = numpy.vstack((data,numpy.hstack((x[:,numpy.newaxis],y[:,numpy.newaxis],a[:,numpy.newaxis]))))
        return data
    
    def normalizeData(self, data):
        edata = data.copy()
        ndata = (edata - numpy.mean(edata,0))/numpy.std(edata,0)
        return ndata

    def compute(self, imgPre, img):
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        allLines = []
        meanvx = 0
        meanvy = 0
        meanx0 = 0
        meany0 = 0
        for c in contours:
            xy = c[:,0,:]
            allLines.append(cv2.fitLine(points=xy, distType=cv2.cv.CV_DIST_L2, param=0, reps=10, aeps=0.01))
        for vx, vy, x0, y0 in allLines:
            meanvx = meanvx + vx
            meanvy = meanvy + vy
            meanx0 = meanx0 + x0
            meany0 = meany0 + y0
        meanvx = meanvx/len(allLines)
        meanvy = meanvy/len(allLines)
        meanx0 = meanx0/len(allLines)
        meany0 = meany0/len(allLines)
        cv2.line(img, (meanx0+self.xoffset, meany0+self.yoffset), (meanx0+self.xoffset+meanvx*100., meany0+self.yoffset+meanvy*100.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        data = self.extractData(contours)
#        for c in contours
#        cv2.line(img, pt1, pt2, color)
        return img#self.normalizeData(data)