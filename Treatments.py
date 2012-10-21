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
        
    def normalizeData(self, data):
        edata = data.copy()
        ndata = (edata - numpy.mean(edata,0))/numpy.std(edata,0)
        return ndata

    def compute(self, imgPre, img):
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        allLines = []
        data = numpy.zeros((len(contours), 2), dtype=numpy.float32)
        datax = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datavx = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datavy = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        i=0
        for c in contours:
            xy = c[:,0,:]
            (vx, vy, x0, y0) = cv2.fitLine(points=xy, distType=cv2.cv.CV_DIST_L2, param=0, reps=10, aeps=0.01)
            data[i, :] = numpy.array([y0[0], numpy.arctan2(vy, vx)[0]], dtype=numpy.float32)
            datax[i] = x0
            datavx[i] = vx
            datavy[i] = vy
            i = i + 1
        retval, bestLabels, centers = cv2.kmeans(data, 4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10) , attempts=cv2.KMEANS_PP_CENTERS, flags=cv2.KMEANS_PP_CENTERS)
        mx0 = numpy.int(numpy.mean((datax)[bestLabels==0]))
        my0 = numpy.int(numpy.mean((data[:, 0:1])[bestLabels==0]))
        mvx0 = numpy.int(numpy.mean((datavx)[bestLabels==0])*100)
        mvy0 = numpy.int(numpy.mean((datavy)[bestLabels==0])*100)
        
        mx1 = numpy.int(numpy.mean((datax)[bestLabels==1]))
        my1 = numpy.int(numpy.mean((data[:, 0:1])[bestLabels==1]))
        mvx1 = numpy.int(numpy.mean((datavx)[bestLabels==1])*100)
        mvy1 = numpy.int(numpy.mean((datavy)[bestLabels==1])*100)
        
        mx2 = numpy.int(numpy.mean((datax)[bestLabels==2]))
        my2 = numpy.int(numpy.mean((data[:, 0:1])[bestLabels==2]))
        mvx2 = numpy.int(numpy.mean((datavx)[bestLabels==2])*100)
        mvy2 = numpy.int(numpy.mean((datavy)[bestLabels==2])*100)
        cv2.line(img, (mx0+self.xoffset, my0+self.yoffset), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        data = self.extractData(contours)
#        for c in contours
#        cv2.line(img, pt1, pt2, color)
        return img#self.normalizeData(data)