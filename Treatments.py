'''
Created on Oct 20, 2012

@author: yvesremi

'''
import cv2, numpy
import skimage.transform
import PreTreatments
from PyQt4 import QtCore
#import cython
#import pyximport; pyximport.install() 

class AbstractTreatment(object):
    def __init__(self):
        '''
        Constructor
        '''
        if(type(self) is AbstractTreatment):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(AbstractTreatment, self).__init__()
        self.filtersToPreApply = [];
        
    def compute(self, img):
            raise NotImplementedError( "The method need to be implemented" )
      
class blobDetectionTreatment(AbstractTreatment):
    
    def __init__(self, mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE, lines=None):
        super(blobDetectionTreatment, self).__init__()
        self.mode = mode
        self.method = method
        self.previousAngle = None
        self.lines = lines
        oldminx = 100000
        oldmaxx = 0
        oldminy = 100000
        oldmaxy = 0
        for line in self.lines:
            newminx = min(line[0].x(), line[1].x())
            oldminx = min(newminx, oldminx)
            newmaxx = max(line[0].x(), line[1].x())
            oldmaxx = max(newmaxx, oldmaxx)
            newminy = min(line[0].y(), line[1].y())
            oldminy = min(newminy, oldminy)
            newmaxy = max(line[0].y(), line[1].y())
            oldmaxy = max(newmaxy, oldmaxy)
        self.limitx = [oldminx, oldmaxx]
        self.limity = [oldminy, oldmaxy]
        self.sortedLines = []
        for line in self.lines:
            self.sortedLines.append((QtCore.QPoint(min(line[0].x(), line[1].x()), min(line[0].y(), line[1].y())),QtCore.QPoint(max(line[0].x(), line[1].x()), max(line[0].y(), line[1].y()))))
        self.xoffset = self.lines[0][0].x()
        self.yoffset =  self.lines[0][0].y()
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([oldminx, oldminy], [oldmaxx, oldmaxy])))
        self.filtersToPreApply.append(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0))
        self.filtersToPreApply.append(PreTreatments.SobelTreatment(dx=1, dy=1, kernelSize=7, scale=1, delta=0))
        self.filtersToPreApply.append(PreTreatments.ThresholdTreatment(-1))
        
        leftpoint0 = numpy.argsort(numpy.asarray([lines[0][0].x(), lines[0][1].x()]))[0]
        leftpoint1 = numpy.argsort(numpy.asarray([lines[1][0].x(), lines[1][1].x()]))[0]
        slope0 = numpy.float(lines[0][1].y()-lines[0][0].y())/numpy.float(lines[0][1].x()-lines[0][0].x())
        slope1 = numpy.float(lines[1][1].y()-lines[1][0].y())/numpy.float(lines[1][1].x()-lines[1][0].x())
        offset0=(lines[0][leftpoint0].y()-oldminy)
        offset1=(lines[1][leftpoint1].y()-oldminy)
        self.limit0 = []
        self.limit1 = []
        for x in range(oldminx, oldmaxx, 1):
            self.limit0.append(slope0*x+offset0) 
            self.limit1.append(slope1*x+offset1)
        
    def compute(self, img):
        imgPre=numpy.copy(img)
        if(self.previousAngle!=None):
            self.filtersToPreApply[1].angleToProcess = [self.previousAngle-5, self.previousAngle, self.previousAngle+5]
        for pretreatments in self.filtersToPreApply:
            imgPre = pretreatments.compute(imgPre)
 
        contours = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)[0]
        datac = numpy.zeros((len(contours), 2), dtype=numpy.float32)
        datas = numpy.ones((len(contours), 2), dtype=numpy.float32)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        datal = numpy.zeros((len(contours), 4), dtype=numpy.float32)
        i=0
        for c in contours:
            if(len(c)>4):
                center, size, angle = cv2.fitEllipse(c)
                datac[i]=center
                datas[i]=size
                dataa[i]=angle
                datal[i] = numpy.squeeze(cv2.fitLine(c, distType = cv2.cv.CV_DIST_HUBER, param = 0, reps = 1, aeps = 1))
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
        datasm = numpy.mean(datas[numpy.all([datas[:, 0]>0], 0)], 0)
        checkAngle=None
        if(self.previousAngle != None):
            checkAngle = ((numpy.abs(dataa[:]-numpy.degrees(self.previousAngle))<10))
        else:
            checkAngle = numpy.asarray(dataa, numpy.bool)
            checkAngle.fill(True)
#        testCenter = [datac[i, 0]>self.limit0[numpy.int(datac[i, 0])] for i in range(datac.shape[0])]
        toKeep = numpy.all([(datas[:, 1]/datas[:, 0])>5.0, (datas[:, 1]*datas[:, 0])>(datasm[0]*datasm[1])*5.0, checkAngle], 0)#(datasm[0]*datasm[1])*7.0)
        dataa = numpy.radians(dataa[toKeep])
        datac=datac[toKeep]
        datas=datas[toKeep]
        mx0 = (self.limitx[1]-self.limitx[0])/2
        my0 = (self.limity[1]-self.limity[0])/2
        newAngle=numpy.float(numpy.median(dataa))
        angle = []
        for values in datal[toKeep]:
            angle.append(numpy.arctan2(values[1], values[0])+numpy.pi/2)   
        newAngle = numpy.float(numpy.median(angle))
#        if(self.previousAngle!=None):
#            if(not numpy.isnan(newAngle)):
#                self.previousAngleSin = numpy.sin(2.0*self.previousAngle/5.0)*numpy.cos(3.0*newAngle/5.0) + numpy.sin(3.0*newAngle/5.0)*numpy.cos(2.0*self.previousAngle/5.0)
##                self.previousAngleCos = numpy.cos(2.0*self.previousAngle/5.0)*numpy.cos(3.0*newAngle/5.0)-numpy.sin(2.0*self.previousAngle/5.0)*numpy.sin(3.0*newAngle/5.0)
#                self.previousAngleCos = numpy.cos(2.0*self.previousAngle/5.0)*numpy.cos(3.0*newAngle/5.0)-numpy.sin(2.0*self.previousAngle/5.0)*numpy.sin(3.0*newAngle/5.0)
#                self.previousAngle = numpy.arctan2(self.previousAngleSin, self.previousAngleCos)
#        else:
        if(not numpy.isnan(newAngle)):
            self.previousAngleSin = numpy.sin(newAngle)
            self.previousAngleCos = numpy.cos(newAngle)
            self.previousAngle = numpy.arctan2(self.previousAngleSin, self.previousAngleCos)
        mvx0 = numpy.int(self.previousAngleSin*5000.0)
        mvy0 = numpy.int(-self.previousAngleCos*5000.0)
        
#        for vx, vy, x0, y0 in zip((numpy.atleast_2d(datavx).T)[bestLabels==0], (numpy.atleast_2d(datavy).T)[bestLabels==0], (numpy.atleast_2d(datax).T)[bestLabels==0], (numpy.atleast_2d(datay).T)[bestLabels==0]):
#            cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

#        for vx, vy, x0, y0 in zip(numpy.sin(dataa), -numpy.cos(dataa), (numpy.atleast_2d(datac[:, 0]).T), (numpy.atleast_2d(datac[:, 1]).T)):
#                        cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(self.previousAngle))[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        for value in range(toKeep.shape[0]):
            if(toKeep[value]):
                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
#            else:
#                cv2.drawContours(img, contours[value], -1, (128, 128, 128), 2, offset=(self.xoffset, self.yoffset))
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
#        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        return (img, self.previousAngle)#self.normalizeData(data)
    
class AponeurosisDetector(AbstractTreatment):
#   
    def __init__(self, mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE, lines=None):
        super(AponeurosisDetector, self).__init__()
        self.mode = mode
        self.method = method
        self.lines = lines
        oldminx = 100000
        oldmaxx = 0
        oldminy = 100000
        oldmaxy = 0
        for line in self.lines:
            newminx = min(line[0].x(), line[1].x())
            oldminx = min(newminx, oldminx)
            newmaxx = max(line[0].x(), line[1].x())
            oldmaxx = max(newmaxx, oldmaxx)
            newminy = min(line[0].y(), line[1].y())
            oldminy = min(newminy, oldminy)
            newmaxy = max(line[0].y(), line[1].y())
            oldmaxy = max(newmaxy, oldmaxy)
        self.limitx = [oldminx, oldmaxx]
        self.limity = [oldminy, oldmaxy]
        self.sortedLines = []
        for line in self.lines:
            self.sortedLines.append((QtCore.QPoint(min(line[0].x(), line[1].x()), min(line[0].y(), line[1].y())),QtCore.QPoint(max(line[0].x(), line[1].x()), max(line[0].y(), line[1].y()))))
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([self.sortedLines[1][0].x(), self.sortedLines[1][0].y()-1], [self.sortedLines[1][1].x(), self.sortedLines[1][1].y()+1])))
        self.angle = numpy.arctan(numpy.float(self.lines[1][1].y()-self.lines[1][0].y())/numpy.float(self.lines[1][1].x()-self.lines[1][0].x())) 
        self.filtersToPreApply.append(PreTreatments.rotateTreatment((self.angle)))
#        self.filtersToPreApply.append(PreTreatments.ReduceSizeTreatment())
        self.filtersToPreApply.append(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0, angleToProcess = [numpy.pi/2.0]))
        self.filtersToPreApply.append(PreTreatments.SobelTreatment(dx=0, dy=1, kernelSize=7, scale=1, delta=0))
        self.filtersToPreApply.append(PreTreatments.ThresholdTreatment(-1))
#        self.filtersToPreApply.append(PreTreatments.addHlineTreatment(thickness = 2, lineDistance=25))
        self.filtersToPreApply.append(PreTreatments.SkeletonTreatment())
#        self.filtersToPreApply.append(PreTreatments.ThinningTreatment())
#        self.filtersToPreApply.append(PreTreatments.erosionTreatment(size=(2, 1)))
        self.filtersToPreApply.append(PreTreatments.DilationTreatment(size=(3, 2)))
#        self.filtersToPreApply.append(PreTreatments.IncreaseSizeTreatment())
        self.xoffset = self.sortedLines[1][0].x()#self.lines[0][0].x()
        self.yoffset =  self.sortedLines[1][0].y()-1#self.lines[0][0].y()
        
#    @jit(argtypes=[uint8[:,:], uint8[:,:]], restype=uint8[:, :])    
    def compute(self, img):
        imgPre = numpy.copy(img) 
        self.filtersToPreApply[1].setAngle(self.angle)
        for pretreatments in self.filtersToPreApply:
            imgPre = pretreatments.compute(imgPre)
        
        contours, hierarchy = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)
        datac = numpy.zeros((len(contours), 2), dtype=numpy.float)
        datas = numpy.ones((len(contours), 2), dtype=numpy.float)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float)
        i=0
        for c in contours:
            if(len(c)>4):
                center, size, angle = cv2.fitEllipse(c)
                datac[i]=center
                datas[i]=size
                dataa[i]=angle
#                cv2.drawContours(img, c, -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
                cv2.ellipse(img, ((center[0]+self.xoffset, center[1]+self.yoffset), size, angle), (255, 255, 255), 1)
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
          
#        goodLines = numpy.argsort((datas[:, 1]))#*datas[:, 1])/datas[:, 0])  
        lineSFactor = datas[:, 1]/max(datas[:, 1])
        lineYFactor = 1.0-numpy.abs(datac[:, 1]-(imgPre.shape[0]/2.0))/imgPre.shape[0]/2.0
        lineAFactor = numpy.squeeze(1.0-numpy.abs(numpy.radians(dataa-90))/numpy.max(numpy.abs(numpy.radians(dataa-90))))
        goodLines = numpy.argsort((numpy.add(numpy.add(lineSFactor, lineYFactor), lineAFactor)))
        for value in goodLines[::-1]:
#            if(datac[value, 1]<=imgPre.shape[1]/4):
            largest = value
            cv2.drawContours(img, contours[value], -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
#                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
            break
        dataa = numpy.radians(dataa-90)
        dataax = numpy.cos(dataa[largest])*numpy.cos(self.angle)-numpy.sin(self.angle)*numpy.sin(dataa[largest])
        dataay = numpy.sin(dataa[largest])*numpy.cos(self.angle)+numpy.sin(self.angle)*numpy.cos(dataa[largest])
        
        dataatot = numpy.arctan(dataay/dataax)
        x0=datac[largest, 0]
        y0=datac[largest, 1]
        x1 = numpy.uint16(-(dataax*500))
        y1 = numpy.uint16(-(dataay*500))
        x2= numpy.uint16(+(dataax*500))
        y2= numpy.uint16(+(dataay*500))
#        cv2.drawContours(img, contours, -1, (255, 255, 255))
        cv2.line(img, (x1+self.xoffset, y1+self.yoffset), (x2+self.xoffset, y2+self.yoffset), (255, 255, 255), 2)
#        cv2.ellipse(img, ((datac[largest][0]+self.xoffset,datac[largest][1]+self.yoffset), datas[largest], -dataa[largest]+90), (255, 255, 255), 2)
        self.angle = dataatot
        if(False):
            for value in goodLines[::-1]:
                if(numpy.abs(numpy.int16(datac[value, 1])-numpy.int16(y0))>imgPre.shape[1]/8):
                    largest = value
                    cv2.drawContours(img, contours[value], -1, (200, 200, 200), 2, offset=(self.xoffset, self.yoffset))
                    break
    #        largest=goodLines[len(goodLines)-2]
            x0=datac[largest, 0]
            y0=datac[largest, 1]
            x1 = numpy.uint16(x0-numpy.sin(dataa[largest])*1000.0)
            y1 = numpy.uint16(y0+numpy.cos(dataa[largest])*1000.0)
            x2= numpy.uint16(x0+numpy.sin(dataa[largest])*1000.0)
            y2= numpy.uint16(y0-numpy.cos(dataa[largest])*1000.0)
    #        cv2.drawContours(img, contours, -1, (255, 255, 255))
    #        cv2.line(img, (x1+self.xoffset, y1+self.yoffset), (x2+self.xoffset, y2+self.yoffset), (255, 255, 255), 2)
    #        cv2.ellipse(img, ((datac[largest][0]+self.xoffset,datac[largest][1]+self.yoffset), datas[largest], -dataa[largest]+90), (255, 255, 255), 2)
        return {img, self.angle}

class AponeurosisTracker(AbstractTreatment):
    
    def __init__(self, lines=None):
        super(AponeurosisTracker, self).__init__()
        self.lines = lines
        self.prevImg = None
        self.sortedLines = (QtCore.QPoint(min(lines[0].x(), lines[1].x()), min(lines[0].y(), lines[1].y())),QtCore.QPoint(max(lines[0].x(), lines[1].x()), max(lines[0].y(), lines[1].y())))
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([self.sortedLines[0].x(), self.sortedLines[0].y()-30], [self.sortedLines[1].x(), self.sortedLines[1].y()+30])))
        self.filtersToPreApply.append(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0))#, angleToProcess = [numpy.pi/2.0]))
        self.xoffset = self.sortedLines[0].x()#self.lines[0][0].x()
        self.yoffset =  self.sortedLines[0].y()-30#self.lines[0][0].y()
        self.prevPts = []
        for percentage in numpy.arange(0, 1.0, 0.05):
            self.prevPts.append((numpy.float32((self.lines[0].x()+(self.lines[1].x()-self.lines[0].x())*percentage)-self.xoffset), numpy.float32((self.lines[0].y()+(self.lines[1].y()-self.lines[0].y())*percentage)-self.yoffset-10)))
            self.prevPts.append((numpy.float32((self.lines[0].x()+(self.lines[1].x()-self.lines[0].x())*percentage)-self.xoffset), numpy.float32((self.lines[0].y()+(self.lines[1].y()-self.lines[0].y())*percentage)-self.yoffset)))
            self.prevPts.append((numpy.float32((self.lines[0].x()+(self.lines[1].x()-self.lines[0].x())*percentage)-self.xoffset), numpy.float32((self.lines[0].y()+(self.lines[1].y()-self.lines[0].y())*percentage)-self.yoffset+10)))
        self.prevPts = numpy.asarray(self.prevPts)
        self.prevImg = None
        
    def compute(self, img):
        imgPre = numpy.copy(img.astype(numpy.uint8)) 

        for pretreatments in self.filtersToPreApply:
            imgPre = pretreatments.compute(imgPre)
        if(self.prevImg == None): 
            self.prevImg = numpy.copy(imgPre)
            self.prevPyramid = cv2.buildOpticalFlowPyramid(img = imgPre, winSize = (21, 21), maxLevel = 3)[1]
        pyramid = cv2.buildOpticalFlowPyramid(imgPre, (21, 21), 3)[1]
        nextPts = cv2.calcOpticalFlowPyrLK(prevImg = self.prevImg, nextImg = imgPre, prevPts = self.prevPts)[0]
        line = cv2.fitLine(nextPts, distType = cv2.cv.CV_DIST_HUBER, param = 0, reps = 1, aeps = 1)
        cv2.line(img, (self.xoffset+line[2]-line[0]*1000.0,self.yoffset+line[3]-line[1]*1000.0), (self.xoffset+line[2]+line[0]*1000.0, self.yoffset+line[3]+line[1]*1000.0), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)                   
        self.prevPyramid = pyramid
        self.prevImg = imgPre
        self.prevPts = nextPts
        angle = numpy.arctan2(line[1], line[0])+numpy.pi/2
#        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(angle))[:6], org=(numpy.uint32(img.shape[0]*0.0), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        return (img, angle)
    
class MuscleTracker(AbstractTreatment):
    
    def __init__(self, lines=None, fiber=None):
        super(MuscleTracker, self).__init__()
        self.previousAngle = None
        self.lines = lines
        oldminx = 100000
        oldmaxx = 0
        oldminy = 100000
        oldmaxy = 0
        for line in self.lines:
            newminx = min(line[0].x(), line[1].x())
            oldminx = min(newminx, oldminx)
            newmaxx = max(line[0].x(), line[1].x())
            oldmaxx = max(newmaxx, oldmaxx)
            newminy = min(line[0].y(), line[1].y())
            oldminy = min(newminy, oldminy)
            newmaxy = max(line[0].y(), line[1].y())
            oldmaxy = max(newmaxy, oldmaxy)
        self.limitx = [oldminx, oldmaxx]
        self.limity = [oldminy, oldmaxy]
        self.sortedLines = []
        for line in self.lines:
            self.sortedLines.append((QtCore.QPoint(min(line[0].x(), line[1].x()), min(line[0].y(), line[1].y())),QtCore.QPoint(max(line[0].x(), line[1].x()), max(line[0].y(), line[1].y()))))
        self.xoffset = self.lines[0][0].x()
        self.yoffset =  self.lines[0][0].y()-30
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([oldminx, oldminy], [oldmaxx, oldmaxy])))
        self.filtersToPreApply.append(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0))
                                      
        self.fiber = fiber
        self.prevImg = None
        leftpointfiber = numpy.argsort(numpy.asarray([fiber[0].x(), fiber[1].x()]))[0]
        distXFiber = fiber[1-leftpointfiber].x()-fiber[leftpointfiber].x()
        distYFiber = fiber[1-leftpointfiber].y()-fiber[leftpointfiber].y()
        self.angle = numpy.pi-numpy.arctan2(distXFiber, distYFiber)
        fiberSlope = numpy.float(fiber[1].y()-fiber[0].y())/numpy.float(fiber[1].x()-fiber[0].x())*(oldmaxx-oldminx)
        offsetFiber=(fiber[leftpointfiber].y()-oldminy)
        
        leftpoint0 = numpy.argsort(numpy.asarray([lines[0][0].x(), lines[0][1].x()]))[0]
        leftpoint1 = numpy.argsort(numpy.asarray([lines[1][0].x(), lines[1][1].x()]))[0]
        slope0 = numpy.float(lines[0][1].y()-lines[0][0].y())/numpy.float(lines[0][1].x()-lines[0][0].x())*(oldmaxx-oldminx)
        slope1 = numpy.float(lines[1][1].y()-lines[1][0].y())/numpy.float(lines[1][1].x()-lines[1][0].x())*(oldmaxx-oldminx)
        offset0=(lines[0][leftpoint0].y()-oldminy)
        offset1=(lines[1][leftpoint1].y()-oldminy)
        
        self.prevPts = []
        for percentagex in numpy.arange(-1.0, 1.0, 0.10):
            x0 = (oldmaxx-oldminx)*percentagex
            y0 = slope0*percentagex+offset0
            goodSlope = (slope1-fiberSlope)/(oldmaxx-oldminx)
            goodOffset= offset1-offset0
            intersectx = -goodOffset/goodSlope + (oldmaxx-oldminx)*percentagex
            intersecty = slope1*intersectx/(oldmaxx-oldminx)+offset1
            x1 = intersectx 
            y1 = slope1*x1/(oldmaxx-oldminx)+offset1
#            y1 = slope1*percentagex+offset1
            x = numpy.uint((percentagex+1.0)*2)
            self.prevPts.append([])
            for percentagey in numpy.arange(0.10, 1.0, 0.10):
                xToAdd = numpy.int((x0+(x1-x0)*percentagey))
                yToAdd = numpy.int((y0+(y1-y0)*percentagey))
                if(xToAdd<(oldmaxx-oldminx) and yToAdd<(oldmaxy-oldminy) and xToAdd>=0 and yToAdd>=0):
                    self.prevPts[x].append((xToAdd, yToAdd))
        self.prevPts = numpy.asarray(self.prevPts)

        self.prevImg = None
        
    def compute(self, img):
        imgPre = numpy.copy(img.astype(numpy.uint8)) 

        for pretreatments in self.filtersToPreApply:
            imgPre = pretreatments.compute(imgPre)
#        imgPre = cv2.GaussianBlur( imgPre, (9, 9), sigmaX=1.0, sigmaY=1.0)
        i=0
        imgPreToPrint = numpy.copy(imgPre)
        for line in self.prevPts:
            for point in line:
                cv2.circle(imgPreToPrint, (point[0], point[1]), 3, (255, 255, 255))
        if(self.prevImg == None): 
            self.prevImg = numpy.copy(imgPre)
            self.prevPyramid = cv2.buildOpticalFlowPyramid(img = imgPre, winSize = (21, 21), maxLevel = 3)[1]
        pyramid = cv2.buildOpticalFlowPyramid(imgPre, (21, 21), 3)[1]
        nextPts = []
        angle = []
        for line in self.prevPts:
            if(numpy.asarray(line).shape[0]>1):
                nextPts.append([])
                nextPts[len(nextPts)-1].append(cv2.calcOpticalFlowPyrLK(prevImg = self.prevImg, nextImg = imgPre, prevPts = numpy.asarray(line, dtype=numpy.float32), winSize=(10, 10))[0])
                linefitted = cv2.fitLine(nextPts[len(nextPts)-1][len(nextPts[len(nextPts)-1])-1], distType = cv2.cv.CV_DIST_HUBER, param = 0, reps = 1, aeps = 1)
                angle.append(numpy.arctan2(linefitted[1], linefitted[0])+numpy.pi/2)                   
        self.prevPyramid = pyramid
        self.prevImg = imgPre
        self.prevPts = numpy.squeeze(numpy.asarray(nextPts))
        i=0
        for value in angle:
            angle[i] = value%(numpy.pi*2.0)
            if(angle[i]>=numpy.pi):
                angle[i] = value-numpy.pi
            i = i+1
        self.angle = numpy.median(angle)
#        finalValueSin = numpy.sin(finalValue)*numpy.cos(self.angle)+numpy.sin(self.angle)*numpy.cos(finalValue)
#        finalValueCos = numpy.cos(finalValue)*numpy.cos(self.angle)-numpy.sin(finalValue)*numpy.sin(self.angle)
#        self.angle = numpy.arctan2(finalValueSin, finalValueCos)
        mx0=(self.limitx[1]-self.limitx[0])/2
        my0=(self.limity[1]-self.limity[0])/2
        mvx0 = numpy.int(numpy.sin(self.angle)*5000.0)
        mvy0 = numpy.int(-numpy.cos(self.angle)*5000.0)
#        self.angle = numpy.median(value)+self.angle
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
#        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(angle))[:6], org=(numpy.uint32(img.shape[0]*0.0), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        return (img, self.angle)
        
class LineFitTreatment(AbstractTreatment):
    def __init__(self, offset = [0, 0], mode = cv2.cv.CV_RETR_LIST, method=cv2.cv.CV_CHAIN_APPROX_NONE):
        self.mode = mode
        self.method = method
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.previousAngle = 2.0
 
    def compute(self, imgPre, img):
        contours = cv2.findContours(imgPre.copy(), mode=self.mode, method=self.method)[0]
        datac = numpy.zeros((len(contours), 2), dtype=numpy.float32)
        datas = numpy.ones((len(contours), 2), dtype=numpy.float32)
        dataa = numpy.zeros((len(contours), 1), dtype=numpy.float32)
        i=0

        for c in contours:
            if(len(c)>4):
                center, size, angle = cv2.fitEllipse(c)
                datac[i]=center
                datas[i]=size
                dataa[i]=angle
#                cv2.ellipse(img, ((center[0]+self.xoffset, center[1]+self.yoffset), size, angle), (255, 255, 255), 1)
                #cv2.line(img, (x0+self.xoffset, y0+self.yoffset), (x0+self.xoffset+vx*20., y0+self.yoffset+vy*20.), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
            i = i + 1
        datasm = numpy.median(datas, 0)

        toKeep = numpy.all([datas[:,1]>datasm[1]*5.0, datas[:, 0]>0.0,(datas[:, 1]/datas[:, 0]>(datasm[1]/datasm[0])*7.0)], 0)
        dataa = numpy.radians(dataa[toKeep])
        datac=datac[toKeep]
        datas=datas[toKeep]

        mx0 = 0
        my0 = 0
        newAngle=numpy.float(numpy.median(dataa))
        if(not numpy.isnan(newAngle)):
            self.previousAngle = 2.0*self.previousAngle/5.0 + 3.0*newAngle/5.0
        mvx0 = numpy.int(numpy.sin(self.previousAngle)*5000.0)
        mvy0 = numpy.int(-numpy.cos(self.previousAngle)*5000.0)
        
#        for vx, vy, x0, y0 in zip((numpy.atleast_2d(datavx).T)[bestLabels==0], (numpy.atleast_2d(datavy).T)[bestLabels==0], (numpy.atleast_2d(datax).T)[bestLabels==0], (numpy.atleast_2d(datay).T)[bestLabels==0]):
#            cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)

#        for vx, vy, x0, y0 in zip(numpy.sin(dataa), -numpy.cos(dataa), (numpy.atleast_2d(datac[:, 0]).T), (numpy.atleast_2d(datac[:, 1]).T)):
#                        cv2.line(img, (numpy.int(x0)+self.xoffset, numpy.int(y0)+self.yoffset), (numpy.int(x0)+self.xoffset+numpy.int(vx*20.), numpy.int(y0)+self.yoffset+numpy.int(vy*20.)), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        cv2.putText(img, "angle = " + numpy.str(numpy.degrees(self.previousAngle))[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        for value in range(toKeep.shape[0]):
            if(toKeep[value]):
                cv2.drawContours(img, contours[value], -1, (255, 255, 255), 2, offset=(self.xoffset, self.yoffset))
#            else:
#                cv2.drawContours(img, contours[value], -1, (128, 128, 128), 2, offset=(self.xoffset, self.yoffset))
        cv2.line(img, (mx0+self.xoffset-mvx0, my0+self.yoffset-mvy0), (mx0+self.xoffset+mvx0, my0+self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
#        cv2.line(img, (mx1+self.xoffset, my1+self.yoffset), (mx1+self.xoffset+mvx1, my1+self.yoffset+mvy1), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
#        cv2.line(img, (mx2+self.xoffset, my2+self.yoffset), (mx2+self.xoffset+mvx2, my2+self.yoffset+mvy2), cv2.cv.Scalar(255, 0, 0), 3, cv2.CV_AA, 0)
        return img


class testRadon(AbstractTreatment):

    def __init__(self, lines=None, manyCircles=False):
        super(testRadon, self).__init__()
        self.previousAngle = None
        self.lines = lines
        self.oldminx = 100000
        self.oldmaxx = 0
        self.oldminy = 100000
        self.oldmaxy = 0
        for line in self.lines:
            newminx = min(line[0].x(), line[1].x())
            self.oldminx = min(newminx, self.oldminx)
            newmaxx = max(line[0].x(), line[1].x())
            self.oldmaxx = max(newmaxx, self.oldmaxx)
            newminy = min(line[0].y(), line[1].y())
            self.oldminy = min(newminy, self.oldminy)
            newmaxy = max(line[0].y(), line[1].y())
            self.oldmaxy = max(newmaxy, self.oldmaxy)
        self.sortedLines = []
        leftpoint0 = numpy.argsort(numpy.asarray([lines[0][0].x(), lines[0][1].x()]))[0]
        leftpoint1 = numpy.argsort(numpy.asarray([lines[1][0].x(), lines[1][1].x()]))[0]
        slope0 = numpy.float(lines[0][1].y()-lines[0][0].y())/numpy.float(lines[0][1].x()-lines[0][0].x())*(self.oldmaxx-self.oldminx)
        slope1 = numpy.float(lines[1][1].y()-lines[1][0].y())/numpy.float(lines[1][1].x()-lines[1][0].x())*(self.oldmaxx-self.oldminx)
        offset0=(lines[0][leftpoint0].y()-self.oldminy)
        offset1=(lines[1][leftpoint1].y()-self.oldminy)
        self.limit0x0 = slope0/4.0+offset0
        self.limit0x1 = slope0/2.0+offset0
        self.limit0x2 = 3.0*slope0/4.0+offset0
        self.limit1x0 = slope1/4.0+offset1
        self.limit1x1 = slope1/2.0+offset1
        self.limit1x2 = 3.0*slope1/4.0+offset1
        self.circlesLimits = [(numpy.int(((self.oldmaxx-self.oldminx)/2.0)), numpy.int(self.limit0x1+numpy.abs(self.limit0x1-self.limit1x1)/2.0), numpy.int(numpy.abs(self.limit0x1-self.limit1x1)/2.0))]
        if(manyCircles==True):
            self.circlesLimits.append((numpy.int(((self.oldmaxx-self.oldminx)/4.0)), numpy.int(self.limit0x0+numpy.abs(self.limit0x0-self.limit1x0)/2.0), numpy.int(numpy.abs(self.limit0x0-self.limit1x0)/2.0)))
            self.circlesLimits.append((numpy.int((3.0*(self.oldmaxx-self.oldminx)/4.0)), numpy.int(self.limit0x2+numpy.abs(self.limit0x2-self.limit1x2)/2.02), numpy.int(numpy.abs(self.limit0x2-self.limit1x2)/2.0)))
        for line in self.lines:
            self.sortedLines.append((QtCore.QPoint(min(line[0].x(), line[1].x()), min(line[0].y(), line[1].y())),QtCore.QPoint(max(line[0].x(), line[1].x()), max(line[0].y(), line[1].y()))))
        self.xoffset = self.oldminx
        self.yoffset =  self.oldminy
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([self.oldminx, self.oldminy], [self.oldmaxx, self.oldmaxy])))
        self.filtersToPreApply.append(PreTreatments.GaborTreatment(ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0))
        self.filtersToPreApply.append(PreTreatments.SobelTreatment(dx=1, dy=1, kernelSize=7, scale=1, delta=0))
        self.filtersToPreApply.append(PreTreatments.ThresholdTreatment(-1))
        self.angle=None
        self.range = 5.0
        self.step = 0.5
        
    def compute(self, img):
        imgPre=numpy.copy(img)
        if(self.angle!=None):
            self.filtersToPreApply[1].angleToProcess = [self.angle-5, self.angle, self.angle+5]
        for pretreatments in self.filtersToPreApply:
            imgPre = pretreatments.compute(imgPre)
        angle = []
        for circleValue in self.circlesLimits:
            mask = numpy.zeros_like(imgPre)
            centerx = circleValue[0]
            centery = circleValue[1]
            radius= circleValue[2]
            cv2.circle(img=mask, center=(centerx, centery), radius=radius, color=255, thickness=-1, lineType=8, shift=0)
            imgPrefiltered = numpy.uint32(numpy.where(mask>=1, imgPre, mask))
            tempResult = self.angle
            dividers = [1.0, 2.0, 4.0]
            if(tempResult==None):
                dividers.append(8.0)
            for i in numpy.asarray(dividers):
                angleRange = numpy.arange(0.0, 180.0, 20)
                if(tempResult!=None):
                    angleRange = numpy.arange(tempResult-(self.range/i), tempResult+(self.range/i), self.step/i) 
                tempResult = angleRange[numpy.argmax(numpy.var(skimage.transform.radon(imgPrefiltered, angleRange), 0))]
                     
            angle.append(numpy.radians(tempResult))
            
            cv2.circle(img=img, center=(numpy.int(self.xoffset+centerx), numpy.int(self.yoffset+centery)), radius=radius, color=255, thickness=1, lineType=8, shift=0)
            imgPrefiltered = cv2.copyMakeBorder(numpy.float32(imgPrefiltered), self.yoffset, img.shape[0]-(imgPrefiltered.shape[0]+self.yoffset), self.xoffset, img.shape[1]-(imgPrefiltered.shape[1]+self.xoffset), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            imgPre2 = cv2.copyMakeBorder(imgPre, self.yoffset, img.shape[0]-(imgPre.shape[0]+self.yoffset), self.xoffset, img.shape[1]-(imgPre.shape[1]+self.xoffset), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img=numpy.where(imgPrefiltered>0, imgPre2, img)
    #        cv2.line(img, (self.xoffset-mvx0, self.yoffset-mvy0), (self.xoffset+mvx0, self.yoffset+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        self.angle = numpy.median(numpy.asarray(angle))
        mvx0 = numpy.int(numpy.sin(self.angle)*5000.0)
        mvy0 = numpy.int(numpy.cos(self.angle)*5000.0)
        cv2.line(img, (numpy.int((self.oldmaxx-self.oldminx)/2.0)+self.oldminx-mvx0, numpy.int((self.oldmaxy-self.oldminy)/2.0)+self.oldminy-mvy0), (numpy.int((self.oldmaxx-self.oldminx)/2.0)+self.oldminx+mvx0, numpy.int((self.oldmaxy-self.oldminy)/2.0)+self.oldminy+mvy0), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        strAngle = numpy.pi/2.0+(numpy.arctan2(numpy.cos(self.angle), numpy.sin(self.angle)))
        self.angle = numpy.degrees(self.angle)
#        cv2.putText(img, "angle = " + numpy.str(strAngle)[:6], org=(numpy.uint32(img.shape[0]*0.75), numpy.uint32(img.shape[1]*0.75)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, bottomLeftOrigin=False)
        return (img, strAngle)
        
class aponeurosisHough(AbstractTreatment):
    
    def __init__(self, offset=[0, 0], angle=0):
        self.xoffset = offset[0]
        self.yoffset = offset[1]
        self.angle = angle
        
        
    def compute(self, imgPre, img):

        center = (imgPre.shape[0]/2.0, imgPre.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, numpy.degrees(self.angle), 1.0)
        alpha = numpy.cos(-self.angle)
        beta = numpy.sin(-self.angle)
        sizemaxX = numpy.int(alpha*imgPre.shape[1])#-beta*img.shape[0])
        sizemaxY = numpy.int(beta*imgPre.shape[1]+alpha*imgPre.shape[0])
        imgPre = cv2.warpAffine(imgPre, M, (sizemaxX, sizemaxY))[sizemaxY/2-30:sizemaxY/2+30, 0:sizemaxX]
        
        lines = cv2.HoughLinesP(image = numpy.copy(numpy.uint8(imgPre)), rho=15, theta=numpy.float(cv2.cv.CV_PI)/180.0, threshold = 2, minLineLength = 100, maxLineGap = 7)
        lines = numpy.squeeze(lines)
        if(len(lines.shape)>1):
            angle = numpy.zeros(lines.shape[0], dtype=numpy.float32)
            leng = numpy.zeros(lines.shape[0], dtype=numpy.float32)
            i=0
            for line in lines:
                leng[i] = numpy.sqrt(numpy.power(line[3]-line[1], 2)+numpy.power(line[2]-line[0], 2))
                if(line[2]-line[0]!=0):
                    angle[i] = numpy.arctan(numpy.float(line[3]-line[1])/numpy.float(line[2]-line[0]))
                elif(line[3]-line[1]>0):
                    angle[i]=cv2.cv.CV_PI/2.0
                else:
                    angle[i]=-cv2.cv.CV_PI/2.0
                i=i+1
            sortedIndex = numpy.argsort(leng)
            n = numpy.ceil(sortedIndex.shape[0]/5.0)
            n = min(sortedIndex.shape[0], 3)
            valueNumber = sortedIndex.shape[0]-n
            accumAngleX = 0
            accumAngleY = 0
            for values in sortedIndex[valueNumber:]:
                accumAngleX = accumAngleX+numpy.cos(angle[values])
                accumAngleY = accumAngleY+numpy.sin(angle[values])
#                accumx1 = accumx1+lines[values, 0]
#                accumy1 = accumy1+lines[values, 1]
#                accumx2 = accumx2+lines[values, 2]
#                accumy2 = accumy2+lines[values, 3]
#                accumAngle = accumAngle+(angle[values]%cv2.cv.CV_PI)
#            accumx1 = accumx1/valueNumber
#            accumy1 = accumy1/valueNumber
#            accumx2 = accumx2/valueNumber*100
#            accumy2 = accumy2/valueNumber*100
            accumAngleX = (accumAngleX/n + numpy.cos(self.angle))*100
            accumAngleY =  (accumAngleY/n + numpy.sin(self.angle))*100
#            accumAngle = (self.angle%cv2.cv.CV_PI + accumAngle/numpy.float(valueNumber))%cv2.cv.CV_PI
            
#            x1 = numpy.uint16(-numpy.cos(accumAngle)*1000.0)
#            y1 = numpy.uint16(-numpy.sin(accumAngle)*1000.0)
#            x2= numpy.uint16(+numpy.cos(accumAngle)*1000.0)
#            y2= numpy.uint16(+numpy.sin(accumAngle)*1000.0)
            cv2.line(img, (self.xoffset, self.yoffset), (self.xoffset+numpy.int(accumAngleX), self.yoffset+numpy.int(accumAngleY)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)    
#            cv2.line(img, (self.xoffset+numpy.int(x1), self.yoffset+numpy.int(y1)), (self.xoffset+numpy.int(x2), self.yoffset+numpy.int(y2)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
    #        cv2.line(img, (self.xoffset+numpy.int(numpy.median(lines[:, 0])), self.yoffset+numpy.int(numpy.median(lines[:, 1]))), (self.xoffset+numpy.int(numpy.median(lines[:, 2])), self.yoffset+numpy.int(numpy.median(lines[:, 3]))), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
            self.angle = numpy.arctan(numpy.float(accumAngleY)/numpy.float(accumAngleX))
        else:
#            leng = numpy.sqrt(numpy.power(lines[3]-lines[1], 2)+numpy.power(lines[2]-lines[0], 2))
            accumAngle = self.angle + numpy.arctan(numpy.float(lines[3]-lines[1])/numpy.float(lines[2]-lines[0]))
            x2= numpy.uint16(numpy.cos(accumAngle)*100.0)
            y2= numpy.uint16(numpy.sin(accumAngle)*100.0)
            cv2.line(img, (self.xoffset, self.yoffset), (self.xoffset+numpy.int(x2), self.yoffset+numpy.int(y2)), cv2.cv.Scalar(255, 0, 0), 2, cv2.CV_AA, 0)
        return img
    
'''
Created on 25 oct 2012

This module has all methods to realize the junction detection on ultrasound images, by seam carving algorithm

@author: Van
'''


class seamCarving(AbstractTreatment):
    def __init__(self, a=5, b=40, hide=False, ref=False):
        super(seamCarving, self).__init__()
        self.a = a
        self.b = b
        self.hide = hide
        self.ref=ref 
        self.xoffset = 50
        self.yoffset = 90
        self.filtersToPreApply.append(PreTreatments.cropTreatment(([self.xoffset, self.yoffset],[565, 505])))
        
    def compute(self, img):
        """
        @param img: matrix of the image
        @param a: thickness of the hiding area (below)
        @param b: thickness of the hiding area (above)
        @param hide: boolean to show or not the hiding area
        @param ref: boolean to show or not a reference point
    
        
        The image for this application needs to be an ultrasound image (in grey).
        The hiding area is the thickness of the fasciae (in pixel). 
        
        make the seams carving process on the image to find the white lines and the junction point
        
        @return the image with the fasciae in blue and the junction in red, the coordinates of the junction
        
        """
        img_orig = numpy.copy(img)
        for pretreatments in self.filtersToPreApply:
            img = pretreatments.compute(img)
        img_copy = numpy.copy(img)
        img_score = self.makeSeams(img_copy)
        coord1 = self.findSeams(img_score, img_score.shape[0], img_score.shape[1])
        
        img_seams2 = self.drawSeamsBlack(img_copy, coord1, self.a, self.b)
        img_score2 = self.makeSeams(img_seams2)
        coord2 = self.findSeams(img_score2, img_score2.shape[0], img_score2.shape[1])
        
        coord1 = self.addOffset(coord1)
        coord2 = self.addOffset(coord2)
        
        junct = self.findJunction(coord1, coord2)
        
        #Draw the 2 seams in blue
        img_fin = self.drawSeamsBlue(img_orig,coord1)
        img_fin = self.drawSeamsBlue(img_fin, coord2)
        
        #Show the hiding thickness in black
        if self.hide == True :
            img_fin = self.drawSeamsBlack(img_fin, coord1, self.a, self.b)
       
        #Reference point
        if self.ref == True :
            img_fin[128:132,128:132] = [50,100,150]   
        
        return img_fin, 0#self.drawPointRed(img_fin, junct), junct
    


    def addOffset(self, pointList):
        """
        @param pointList: List of the points for which the offset will be added
        
        @return the offset list
        
        """
        for point in pointList:
            point[0] = point[0]+self.yoffset;
            point[1] = point[1]+self.xoffset;
        return pointList
            
    def makeSeams(self, img):
        """
        @param img: Matrix of the image
        
        @return the array of the Seams Carving Image
        
        """
                
        dimr = img.shape[0] #rows
        dimc = img.shape[1] #columns
        
        #Image inversee completement
        #img_rev = - img[:,-1::-1] 
        
        img_new = numpy.ones(img.shape)
        img_new = img
        
        img_new[img_new[:,:] == 0] = 1
         
        #Image avec couleur inversee
        img_rev= -img_new
        
       
        #print dimr
        #print dimc 
      
           
        return self.pixelScore(img_rev, dimr, dimc)
    
    
    
    def pixelScore(self, img, dimr, dimc):
        """
        @param img: matrix of the image
        @param dimr: dimension for rows
        @param dimc: dimension for columns
        
        @return matrix of score following horizontal seam carving algorithm
        """
           
        score = numpy.zeros_like(img, dtype = float) 
        #Copy of the last column
        score[:,dimc-1] = img[:, dimc-1] 
        
        #Beginning 
        for j in range(dimc-2, -1, -1):
            for i in range(dimr-1, -1, -1):
                #If the right border is reached, only 2 values are tested. to find the minimum
                if(i == dimr-1): 
                    score[i,j] = img[i, j] + min([score[i,j+1], score[i-1,j+1]]) 
                #If the left border is reached, only 2 values are tested 
                elif(i == 0): 
                    score[i,j] = img[i, j] + min([score[i,j+1], score[i+1,j+1]]) 
                #Else, 3 values are tested
                else: 
                    score[i,j] = img[i, j] + min([score[i,j+1],score[i+1,j+1], score[i-1,j+1]]) 
                
                 
        #plt.imshow(score)
        #plt.colorbar()
        #plt.gca().set_ylim(0,score.shape[0])
        #plt.show() 
        
        return score 
    
       
    def findSeams(self, img_score, dimr, dimc):
        """
        @param img_score: matrix of score of the image
        @param dimr: dimension for the rows
        @param dimc: dimension for the columns
        
        @return the coordinates of the seams (way with the lowest cost)
        
        """ 
        coord=[]
#!!!        
        i = numpy.atleast_1d((numpy.argmin(img_score[:,0], axis=0)))
        i = i[0]
        
        self.findSeamsHor(coord, i, img_score, dimr, dimc)
               
        return coord
    
    def findSeamsHor(self, coord, i, img_score, dimr, dimc):
        """
        @param coord: list of the coordinates of the seams
        @param i: arguments of the row with minimum value
        @param img_score: matrix of the image score
        @param dimr: dimension of rows
        @param dimc: dimension of columns
        
        add coordinates of the seams in a list, by finding the minimum value of the adjacent rows of the next column
        
        """
        
        for j in range(dimc-1):
            if i <= dimr-2:
                coord.append([i,j])
                if img_score[i+1,j]<= img_score[i,j] and  img_score[i+1,j]<= img_score[i-1,j]:
                    i = i+1
                elif img_score[i,j]<= img_score[i+1,j] and  img_score[i,j]<= img_score[i-1,j]:
                    i = i
                elif img_score[i-1,j]<= img_score[i,j] and  img_score[i-1,j]<= img_score[i+1,j]:
                    i = i-1
            elif i == dimr-1:
                coord.append([i,j])
                if img_score[i-1,j] < img_score[i,j]:
                    i=i-1
            else:
                coord.append([i,j])
                if img_score[i+1,j] < img_score[i,j]:
                    i=i+1    
    
    
    def drawSeamsBlue(self, img, coord):
        """
        @param img: matrix of the image
        @param coord: array of the coordinates [x,y] of the seams
        
        Draw the image with seams in blue
        
        @return the image with the seams in blue
        
        """
        for i in coord :
            x = i[0]
            y = i[1]
            img[x,y] = 255
            
        return img
        
        
    def drawSeamsBlack(self, img, coord, a=5, b=40):
        #images jpeg a=5 et b = 40
        """
        @param img: matrix of the image
        @param coord: array of the coordinates [x,y] of the seams
        @param a: thickness of the hiding area below
        @param b: thickness of the hiding area above
        
        Draw the image with some thickness in black following the first seam
        
        @return the image with the seams in black
        """
        for i in coord :
            x = i[0]
            y = i[1]
            
            
            img[x-a:x+b] = 0
      
        return img
    
    def drawPointRed(self, img, point):
        """
        @param img: matrix of the image
        @param point: coordinate of the point of junction
        
        Draw a red little square at the position given by "point"
        
        return the image with the red point
        
        """
        if point != [] :
            r = point[0]
            c = point[1]
            if r>1 and r<img.shape[0] -1 and c>1 and c< img.shape[1] - 1 :
                img[r-2:r+2, c-2:c+2] = 255
        
        return img
    
    
    def findJunction(self, coord1, coord2):
        """
        @param coord1: coordinates of the first fascia
        @param coord2: coordinates of the second fascia
        
        @return the coordinate of the junction
        
        """
        junct= []
        
        
        #If fascia from above is detected first
        if (coord1[0][0] - coord2[0][0] > 0) : 
            junct = [(coord1[0][0] + coord2[0][0])/2,0]
            diff = coord1[0][0] - coord2[0][0]
            for i in range(len(coord1)):
    
                if numpy.abs(coord1[i][0] - coord2[i][0]) < diff : 
                    diff = coord1[i][0] - coord2[i][0]
                    junct = [(coord1[i][0] + coord2[i][0])/2, i]
                    
        # if fascia from below is detected first
        else : 
            junct = [(coord2[0][0] + coord1[0][0])/2,0]
            diff = coord2[0][0] - coord1[0][0]
            for i in range(len(coord2)):
    
                if numpy.abs(coord2[i][0] - coord1[i][0]) < diff : 
                    diff = coord2[i][0] - coord1[i][0]
                    junct = [(coord2[i][0] + coord1[i][0])/2, i]         
                      
        
        return junct