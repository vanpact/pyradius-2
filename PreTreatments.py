#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: PreTreatments
        :platform: Unix, Windows
        :synopsis: module which provide the preTreatments for the video.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""

from PyQt4 import QtCore, QtGui, QtMultimedia
import cv2, numpy, sys
from VideoWidget import Movie
from multiprocessing.pool import ThreadPool
import multiprocessing
import math, time
from imageConverter import ImageConverter
from threading import Thread
import threading 
from Treatments import AbstractTreatment
import PostTreatments
import cython
import pyximport; pyximport.install() 

class computationType:     
    pennationAngleComputation = 0;
    JunctionComputation = 1;
    
class Applier(QtCore.QThread):
    """This is the class which apply all the filters. For the preTreatment and the treatments."""
    frameComputed = QtCore.pyqtSignal()
    __single = None
    def __init__(self):
        """Constructor of the class.
        
        Kwargs:
            src: the video source on which the filters will be applied.
        
         Raises:
             TypeError: if the type of the source is neither a Movie nor None.
        """
        if(Applier.__single is None):
            super(Applier, self).__init__()
            Applier.__single = self 
        Applier.__single.nrChannel=None
#        Applier.__single.filtersToPreApply = None
#        Applier.__single.filtersToPreApply = None
        Applier.__single.filtersToApply = None
        Applier.__single.filtersToApply = None
        Applier.__single.processedVideo = None
        Applier.__single.lastComputedFrame = None
        Applier.__single.wait=True
        Applier.__single.src = None
        Applier.__single.angleFinal=0
        Applier.__single.nrSkipFrame = 0
        Applier.__single.computationType = computationType.pennationAngleComputation
        Applier.__single.writeResults=False
        Applier.__single.writeFile='output.txt'
        
    @staticmethod
    def getInstance(src=None, nrChannel=1, nrSkipFrame=0, computType = computationType.pennationAngleComputation):
        if(Applier.__single is None):
            Applier.__single = Applier()
#            Applier.__single = Applier.__single
        Applier.__single.nrChannel=nrChannel
#        Applier.__single.filtersToPreApply = []
#        Applier.__single.filtersToPreApply = [ [] for i in range(nrChannel)]
        Applier.__single.filtersToApply = []
        Applier.__single.filtersToApply = [[] for i in range(nrChannel)]
        Applier.__single.processedVideo = []
        Applier.__single.lastComputedFrame = None
        Applier.__single.wait=True
        Applier.__single.angleFinal=0
        Applier.__single.nrSkipFrame=nrSkipFrame
        Applier.__single.computationType = computType
        Applier.__single.writeResults=False
        Applier.__single.writeFile='output.txt'
        if(isinstance(src, Movie)):
            Applier.__single.src = src
        elif src is not None:
            raise TypeError("Src must be None or a Movie!") 
        return Applier.__single
        
    def toggle(self):
        """Toggle the applier state. if wait is true, the applier will pause the processing of the video."""
        self.wait = not self.wait
        
    def setWriteResults(self, isWriting, fileObject):
        if(isinstance(fileObject, file)):
            self.writeResult = isWriting
            self.writeFile = fileObject
        else:
            self.writeResult = False
            self.writeFile = None
        
    def setSource(self, src=file):
        if(isinstance(src, Movie)):
            self.wait=True
            self.processedVideo = []
            self.src = src
        else:
            raise TypeError("Src must be none or a Movie!")
        
    def add(self, other, channel=0):
        if(isinstance(other, AbstractTreatment)):
            self.filtersToApply[channel].append(other)
            return self
        else:
            raise TypeError("Object send to the applier has to be a pretreatment.")
    
    def sub(self, other, channel=0):
        if(isinstance(other, AbstractTreatment)):
            self.filtersToApply[channel].reverse()
            self.filtersToApply[channel].remove(other)
            self.filtersToApply[channel].reverse()
            return self
        else:
            raise TypeError("Object send to the applier has to be a pretreatment.")
        
    def empty(self):
        self.filterToPreApply = []
        
    def apply(self, img):
        if(len(img.shape)>2):
            img = img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.114
        imgToMix=[numpy.copy(img) for _ in range(self.nrChannel)]
        angle = [0 for _ in range(self.nrChannel)]
        for channel in range(self.nrChannel):
            for treatment in self.filtersToApply[channel]:
                imgToMix[channel], angle[channel] = treatment.compute(imgToMix[channel])
        for imgProcessed in imgToMix:
            img = numpy.maximum(img, imgProcessed)
            
        if(self.computationType == computationType.pennationAngleComputation):
            self.angleFinal = PostTreatments.computeAngle(angle[1:3], angle[0])
        if(self.writeResult):
            self.writeFile.write(str(self.angleFinal)+ '\n') 
#        if(self.computationType==computationType.JunctionComputation):
#            self.AngleFinal = PostTreatments.computeAngle(angle[1:3], angle[0])
#        img=imgPre
        return img
    
    def setComputationType(self, type):
        self.computationType = type
            
    def applyNext(self):
        self.src.readNCFrame(self.nrSkipFrame)
        ndimg = self.src.rawBuffer
        ndimg = self.apply(ndimg)
        self.lastComputedFrame=ndimg
#        self.processedVideo.append(ndimg)
        self.frameComputed.emit()
        return QtMultimedia.QVideoFrame(ImageConverter.ndarrayToQimage(ndimg))
    
    def applyOne(self):
        self.src.readNCFrame(self.nrSkipFrame)
        ndimg = self.src.rawBuffer
        if(len(ndimg.shape)>2):
            ndimg = ndimg[:, :, 0]*0.299+ndimg[:, :, 1]*0.587+ndimg[:, :, 2]*0.114
        self.lastComputedFrame=ndimg
#        self.processedVideo.append(ndimg)
        self.frameComputed.emit()
        return QtMultimedia.QVideoFrame(ImageConverter.ndarrayToQimage(ndimg))
        
    def applyAll(self):
        frameNb = self.src.getFrameNumber()
        for i in range(0, frameNb, self.nrSkipFrame+1):
            self.applyNext()
            while(self.wait):
                self.msleep(50)
    
    def run(self):
        self.applyAll()
        
    def getLastComputedFrame(self):
        frameToSend = ImageConverter.ndarrayToQimage(self.lastComputedFrame).copy()#self.processedVideo[-1]).copy()
        return QtMultimedia.QVideoFrame(frameToSend)
    
    def getLastComputedAngle(self):
        return self.angleFinal
    
    def saveResult(self, fileName):
        if(len(self.processedVideo)>0 and (fileName[-4:]=='.avi' or fileName[-4:]=='.AVI')):
            videoWriter = cv2.VideoWriter(fileName, 0, 25, (self.processedVideo[0].shape[0], self.processedVideo[0].shape[1]), isColor=False)
            for frame in self.processedVideo:
                videoWriter.write(frame)
        else: 
            raise ValueError("You have to use the applier on each frame you want to process and the file name has to finish by '.avi'.")
            
class AbstractPreTreatment(AbstractTreatment):
    
    def __init__(self):
        '''
        Constructor
        '''
        if(type(self) is AbstractPreTreatment):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(AbstractPreTreatment, self).__init__()

    
class CannyTreatment(AbstractPreTreatment):
    '''
    classdocs
    '''
    minThreshold = 25
    ratio = 3
    kernelSize = 3
    
    def __init__(self, minThreshold = 200, ratio = 1.2, kernelSize = 7, L2gradient=False):
        '''
        Constructor
        '''
        super(CannyTreatment, self).__init__()
        self.minThreshold = minThreshold
        self.ratio = ratio
        self.kernelSize = kernelSize
        self.L2gradient = L2gradient
        
    def compute(self, img):
        if(len(img.shape)==2):
            img= cv2.GaussianBlur( img, (9, 9), sigmaX=1.0, sigmaY=1.0)
    #        image = cv2.cvtColor( image, cv2.cv.CV_RGB2GRAY)
#            img = cv2.blur(img, self.kernelSize)
#            edges = cv2.medianBlur(numpy.uint8(img), 3)
            edges = cv2.Canny(numpy.uint8(img), self.minThreshold, self.minThreshold*self.ratio, apertureSize=self.kernelSize, L2gradient=self.L2gradient)#self.kernelSize )
#            result = numpy.zeros( img.shape, img.dtype )
    #        image.copyTo(result, edges)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
        return edges

        
class GaborTreatment(AbstractPreTreatment):
    filters = []
    
    def __init__(self, ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0, ktype = numpy.float32, angleToProcess=[]):
        super(GaborTreatment, self).__init__()
        self.angleToProcess = angleToProcess
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.ktype = ktype
        self.cpuNumber = 1#max(multiprocessing.cpu_count()-2, 1)
        self.pool = ThreadPool(processes=self.cpuNumber)
        self.buildFilters()
       
    def getGaborKernel(self, theta): 
        sigma_x = self.sigma
        sigma_y = self.sigma/self.gamma
        nstds = 3
        xmin=0 
        xmax=0
        ymin=0
        ymax=0
        c = math.cos(theta)
        s = math.sin(theta)
        if( self.ksize > 0 ):
            xmax = self.ksize/2
        else:
            xmax = max(math.fabs(nstds*sigma_x*c), math.fabs(nstds*sigma_y*s))
        ymax = xmax
        ymin = -ymax
        xmin=-xmax
#        if( self.ksize > 0 ):
#            ymax = self.ksize/2
#        else:
#            ymax = max(math.fabs(nstds*sigma_x*c), math.fabs(nstds*sigma_y*s))
        kernel = numpy.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype = self.ktype)
        scale = 1.0#1/(2*numpy.pi*sigma_x*sigma_y)
        ex = -0.5/(sigma_x*sigma_x)
        ey = -0.5/(sigma_y*sigma_y)
        cscale = numpy.pi*2/self.lambd
        
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                xr = x*c + y*s
                yr = -x*s + y*c
#                v = scale*math.exp(ex*xr*xr + ey*yr*yr)*math.cos(cscale*xr + self.psi)
                v = scale*math.exp(ex*xr*xr + ey*yr*yr)*math.cos(cscale*xr + self.psi)
                kernel[ymax + y, xmax + x] = v
        return kernel
    
    def buildFilters(self):
        angles = 0
        if(self.angleToProcess==[]):
            self.angleToProcess = numpy.arange(0, numpy.pi, numpy.pi / 32)
        for theta in self.angleToProcess:
            params = {'ksize':(self.ksize, self.ksize), 'sigma':self.sigma, 'theta':theta, 'lambd':self.lambd,
                      'gamma':self.gamma, 'psi':self.psi, 'ktype':numpy.float32}
            kern = self.getGaborKernel(theta)
            ks=kern.sum()
            if(ks >0.01):
                kern = kern/(ks)
            self.filters.append((kern, params))
        return self.filters
    
    def process(self, img):
        accum = numpy.zeros_like(img)
        angle = numpy.zeros_like(img,dtype=numpy.float32)
        for kern, params in self.filters:
#            fimg = numpy.zeros_like(img, img.dtype)
#            kern = numpy.ones(kern.shape, kern.dtype)
#            cv2.cv.Filter2D(cv2.cv.fromarray(img), cv2.cv.fromarray(fimg), cv2.cv.fromarray(kern))
#            if(len(img.shape)>2):
#                for d in range(0, img.shape[2]):
#                    tmp = numpy.copy(img[:, :, d])
#                    fimg[:, :, d] = cv2.filter2D(tmp, cv2.CV_8U, kern)
#            elif(len(img.shape)==2):
            fimg = cv2.filter2D(img[:, :], cv2.CV_8U, kern)
            
#            else:
#                raise ValueError("The image must have 2 or 3 dimension.")
            numpy.maximum(accum, fimg, accum)
#            angle[numpy.equal(fimg,accum)] = params['theta']
        return accum

    def processThreaded(self, img):
        accum = numpy.zeros_like(img)
#        angle = numpy.zeros_like(img,dtype=numpy.float32)
        accumLock = multiprocessing.Lock()
        idx=None
        def f(filt):
            kern, params = filt
#            fimg = numpy.zeros_like(img, img.dtype)
#            if(len(img.shape)>2):
#                for d in range(0, img.shape[2]):
#                    tmp = numpy.copy(img[:, :, d])
#                    fimg[:, :, d] = cv2.filter2D(tmp, cv2.CV_8U, kern)
#            elif(len(img.shape)==2):
            fimg = cv2.filter2D(img[:, :], cv2.CV_8U, kern)
#            else:
#                raise ValueError("The image must have 2 or 3 dimension.")
            with accumLock:
#                if(len(img.shape)>2):
#                    for d in range(0, img.shape[2]):
#                        idx = numpy.argmax(numpy.dstack((accum[:, :, d], fimg[:, :, d])),axis=2)
#                        accum[:, :, d][idx==1]=fimg[:, :, d][idx==1]
#                        angle[:, :, d][idx==1]=numpy.sin(params['theta'])*100
#                elif(len(img.shape)==2):
                idx = numpy.argmax(numpy.dstack((accum, fimg)),axis=2)
                accum[idx==1]=fimg[idx==1]
#            angle[idx==1]=numpy.sin(params['theta'])*100
#                else:
#                    raise ValueError("The image must have 2 or 3 dimension.")
    
        
        self.pool.map(f,self.filters)
        return accum
    
    def compute(self, img):
        if(len(img.shape)==2):
            if(self.cpuNumber>1):
                img = self.processThreaded(numpy.copy(img))
            else:
                img = self.process(numpy.copy(img))
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
        return img

class ReduceSizeTreatment(AbstractPreTreatment):
    
    def __init__(self, dstSize=None):
        super(ReduceSizeTreatment, self).__init__()
        self.dstSize=dstSize
        
    def compute(self, img):
        if(len(img.shape)==2):
                return cv2.pyrDown(img, dstsize=self.dstSize)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class IncreaseSizeTreatment(AbstractPreTreatment):
    
    def __init__(self, dstSize=None):
        super(IncreaseSizeTreatment, self).__init__()
        self.dstSize=dstSize
    
    def compute(self, img):
        if(len(img.shape)==2):
                return numpy.uint8(cv2.pyrUp(img, dstsize=self.dstSize))
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class changeContrastTreatment(AbstractPreTreatment):
    
    def __init__(self, value=1.0):
        super(changeContrastTreatment, self).__init__()
        self.value=value
    
    def compute(self, img):
        image = numpy.zeros(img.shape, dtype = numpy.uint32)
        if(len(img.shape)==2):
                image = numpy.uint32(img)*self.value
                image[image >255] = 255
                return numpy.uint8(image)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class cropTreatment(AbstractPreTreatment):

    def __init__(self, roi = None):
        super(cropTreatment, self).__init__()        
        self.roi = roi
    
    def compute(self, img):
        if(self.roi is not None and self.roi[0][0]>=0 and self.roi[0][1]>=0 and self.roi[0][0]<self.roi[1][0] and self.roi[0][1]<self.roi[1][1] and self.roi[1][0]<img.shape[0] and self.roi[1][1]<img.shape[1]):
            return img[self.roi[0][1]:self.roi[1][1],self.roi[0][0]:self.roi[1][0]].copy()
        else: 
            raise ValueError("Incorrect size for the region of interest when cropping.")    

class rotateTreatment(AbstractPreTreatment):
    def __init__(self, angle = 0):
        super(rotateTreatment, self).__init__()
        self.setAngle(angle)
    
    def setAngle(self, angle):
        self.angle=angle
        self.alpha = numpy.cos(-self.angle)
        self.beta = numpy.sin(-self.angle)
        
    def compute(self, img):
        center = (img.shape[0]/2.0, img.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, numpy.degrees(self.angle), 1.0)
        sizemaxX = numpy.int(numpy.abs(self.alpha*img.shape[1]))#-beta*img.shape[0])
        sizemaxY = numpy.int(numpy.abs(self.beta*img.shape[1]+self.alpha*img.shape[0]))
        return cv2.warpAffine(img, M, (sizemaxX, sizemaxY), borderMode=cv2.BORDER_WRAP).copy()#[sizemaxY/2-30:sizemaxY/2+30, 0:sizemaxX]
        
class LaplacianTreatment(AbstractPreTreatment):

    def __init__(self, kernelSize=3, scale=1, delta=0):
        super(LaplacianTreatment, self).__init__()  
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
        img= cv2.GaussianBlur( img, (9, 9), sigmaX=0, sigmaY=0)
        result = cv2.Laplacian( numpy.uint8(img), ddepth=cv2.CV_32F, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
#        result = -result
        result[result<0]=0
        return numpy.uint8((result/numpy.max(result))*255.0)
#        result[result<50]=0
    
class SobelTreatment(AbstractPreTreatment):

    def __init__(self, dx=1, dy=1, kernelSize=7, scale=1, delta=0):
        super(SobelTreatment, self).__init__()  
        self.dx=dx
        self.dy=dy
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
#        img= cv2.GaussianBlur( img, (31,31), sigmaX=1.0, sigmaY=1.0)
        result = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=self.dx, dy=self.dy, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        result[result<0] = 0
        result = result - numpy.min(result)
        return numpy.uint8((result/numpy.max(result))*255.0)
#        result = cv2.convertScaleAbs( result)
#        result[result<50]=0
        return result
    
    def compute2(self, img):
#        img = cv2.GaussianBlur( img, (7, 7), sigmaX=0.0, sigmaY=0.0);
        grad_x = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7, scale=1, delta=0)
#        abs_grad_x = cv2.convertScaleAbs( grad_x,  alpha=255.0/numpy.max(numpy.abs(grad_x)))
        
        grad_y = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=0, dy=1, ksize=7, scale=1, delta=0)
#        abs_grad_y = cv2.convertScaleAbs( grad_y,  alpha=255.0/numpy.max(numpy.abs(grad_y)))
        norm =  numpy.sqrt(numpy.add(numpy.square(numpy.asarray(grad_x, dtype=numpy.float)), numpy.square(numpy.asarray(grad_y, dtype=numpy.float))))#cv2.addWeighted( numpy.abs(grad_x), 0.5, numpy.abs(grad_y), 0.5, 0)
        return numpy.asarray((norm/numpy.max(norm))*255.0, dtype=numpy.uint8)
class DOHTreatment(AbstractPreTreatment):

    def __init__(self, kernelSize=7, scale=1, delta=0):
        super(DOHTreatment, self).__init__()  
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
#        img= cv2.GaussianBlur( img, (9, 9), sigmaX=0, sigmaY=0)
        resultxy = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=1, dy=1, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        resultxx = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=2, dy=0, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        resultyy = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=0, dy=2, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        result = resultxx*resultyy-resultxy*resultxy
        
        result[not numpy.all((result>0, resultxx<=0))]=0
        result[numpy.all((result>0, resultxx<=0))]=255
#        result[result<numpy.median(result)]=0
#        result = result-numpy.median(result)
        return numpy.uint8((result/numpy.max(result))*255.0)
#        result[result<50]=0
   
class rotationTreatment(AbstractPreTreatment):
    def __init__(self, angle):
        super(rotationTreatment, self).__init__()
        self.angle = angle
        
    def compute(self, img):
        center = (img.shape[0]/2.0, img.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        alpha = numpy.cos(-numpy.radians(self.angle))
        beta = numpy.sin(-numpy.radians(self.angle))
        sizemaxX = numpy.int(alpha*img.shape[1])#-beta*img.shape[0])
        sizemaxY = numpy.int(beta*img.shape[1]+alpha*img.shape[0])
        result = cv2.warpAffine(img, M, (sizemaxX, sizemaxY))
        return result[result.shape[0]/2-30:result.shape[0]/2+30, 0:result.shape[1]].copy()
    
class ThresholdTreatment(AbstractPreTreatment):
    def __init__(self, threshold=-1):
        super(ThresholdTreatment, self).__init__()
        self.threshold = threshold
        
    def compute(self, img):
        if(self.threshold<0):
            self.threshold=numpy.median(img[img>numpy.max(numpy.min(img), 0)])
        if(self.threshold<-1):
            self.threshold=numpy.median(img[img>=numpy.max(self.threshold, 0)])
        img[img<self.threshold]=0
        img[img>=self.threshold]=255
        return img
    
class addHlineTreatment(AbstractPreTreatment):
    def __init__(self, thickness = 1, lineDistance=10):
        super(addHlineTreatment, self).__init__()
        self.thickness = thickness
        self.lineDistance = numpy.int(lineDistance)
        
    def compute(self, img):
        x = img.shape[1]
        y = img.shape[0]
#        lineNumber = numpy.int(y/self.lineDistance)
        for i in numpy.arange(self.lineDistance-1,y,self.lineDistance):
            cv2.rectangle(img, (0, i), (x, i+self.thickness), (0, 0, 0), thickness=-1)
        return img.copy()
        
class DilationTreatment(AbstractPreTreatment):
    def __init__(self, morphology=cv2.MORPH_RECT, size=(3, 3)):
        super(DilationTreatment, self).__init__()
        self.morphology=morphology
        self.size=size
        
    def compute(self, img):
        elem = cv2.getStructuringElement( self.morphology, self.size )
        return cv2.dilate(img, elem)#cv2.dilate(cv2.erode(img, elem), elem)

class erosionTreatment(AbstractPreTreatment):
    def __init__(self, morphology=cv2.MORPH_RECT, size=(3, 3)):
        super(erosionTreatment, self).__init__()
        self.morphology=morphology
        self.size=size
        
    def compute(self, img):
        elem = cv2.getStructuringElement( self.morphology, self.size )
        return cv2.erode(img, elem)#cv2.dilate(cv2.erode(img, elem), elem)

class SkeletonTreatment(AbstractTreatment):
    def __init__(self, size=(3, 3)):
        super(SkeletonTreatment, self).__init__()
        self.size=size
        
    def compute(self, img):

        k = cv2.getStructuringElement( cv2.MORPH_CROSS, self.size )
        skel = numpy.zeros_like(img, dtype = numpy.uint8)
        temp = numpy.zeros_like(img, dtype = numpy.uint8)
        done = False
        while(not done):
            temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
            temp = numpy.logical_not(temp)
            temp = numpy.logical_and(img, temp)
            skel = numpy.logical_or(skel, temp)
            img = cv2.erode(img, k)
            done = not numpy.any(img)
#        for i in numpy.arange(0, 20, 1):
#            elem = numpy.array([(-1, -1, -1),(0, 1, 0),(1, 1, 1)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(0, -1, -1),(1, 1, -1),(1, 1, 0)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(-1, 0, 1),(-1, 1, 1),(-1, 0, 1)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(-1, -1, 0),(-1, 1, 1),(0, 1, 0)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(1, 1, 1),(0, 1, 0),(-1, -1, -1)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(0, 1, 0),(-1, 1, 1),(-1, -1, 0)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(1, 0, -1),(1, 1, -1),(1, 0, -1)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
#            elem = numpy.array([(0, 1, 0),(1, 1, -1),(0, -1, -1)], dtype=numpy.int8)
#            img = self.erodeSP(img, elem)
        return skel.astype(numpy.uint8)*255
    def erodeSP(self, img, elem):
        imgPre = numpy.copy(img)
        for y in numpy.arange(1, img.shape[0]-2, 1):
            for x in (1, img.shape[1]-2, 1):
                tmp = numpy.copy(img[max(y-numpy.int(elem.shape[0]/2), 0):min(y+numpy.int(elem.shape[0]/2)+1, img.shape[0]), max(x-numpy.int(elem.shape[1]/2), 0):min(x+numpy.int(elem.shape[1]/2)+1, img.shape[1])]).astype(numpy.int)
                tmp[numpy.where(elem==-1)] = -1
                tmp[numpy.where(tmp>0)] = 1
                if(numpy.all(elem == tmp) and tmp[1][1]!=1 or (not(numpy.all(elem == tmp)) and tmp[1][1]==1)):
                    a=1
                imgPre[y][x] = 255 if (numpy.all(elem == tmp)) else 0
        return imgPre
    
class ThinningTreatment(AbstractTreatment):
    def __init__(self, size=(3, 3)):
        super(ThinningTreatment, self).__init__()
        self.size=size
#        self.time1 = []
#        self.time2 = []
#        self.time3 = []
#        self.time4 = []
#        self.time5 = []
#        self.time6 = []
    def thinningIteration(self, img, iter):
#        cdef int v2
#        cdef int v3
#        cdef int v4
#        cdef int v5
#        cdef int v6
#        cdef int v7
#        cdef int v8
#        cdef int v9
#        cdef int m1
#        cdef int m2
#        cdef int A
#        cdef int B
        marker = numpy.zeros_like(img, dtype = numpy.uint8)
        for y in numpy.arange(1, img.shape[0]-2, 1):
            for x in numpy.arange(1, img.shape[1]-2, 1):
#                t0= time.clock()
#                tmp = numpy.copy(img[y-1:y+2, x-1:x+2])
#                self.time6.append(time.clock()-t0)
#                t0= time.clock()
                v2 = img[y-1, x]
                v3 = img[y-1, x+1]
                v4 = img[y, x+1]
                v5 = img[y+1, x+1]
                v6 = img[y+1, x]
                v7 = img[y+1, x-1]
                v8 = img[y, x-1]
                v9 = img[y-1, x-1]
#                self.time1.append(time.clock()-t0)
#                t0= time.clock()
                B = v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
#                self.time2.append(time.clock()-t0)
#                t0= time.clock()
                if(B >= 2 and B <= 6):
#                    self.time5.append(time.clock()-t0)
                    if(iter == 0):
#                        t0= time.clock()
                        m1 = (v2 * v4 * v6)
                        m2 = (v4 * v6 * v8)
#                        self.time3.append(time.clock()-t0)
                    else :
#                        t0= time.clock()
                        m1 = (v2 * v4 * v8)
                        m2 = (v2 * v6 * v8)
#                        self.time3.append(time.clock()-t0)
    #                m1 = (tmp[0][1] * tmp[1][2] * tmp[2][1]) if(iter == 0) else (tmp[0][1] * tmp[1][2] * tmp[1][0])
    #                m2 = (tmp[1][2] * tmp[2][1] * tmp[1][0]) if(iter == 0) else (tmp[0][1] * tmp[2][1] * tmp[1][0])
                    if( m1 == 0 and m2 == 0):
#                        t0= time.clock()
                        v2 = v2==0
                        v3 = v3==0
                        v4 = v4==0
                        v5 = v5==0
                        v6 = v6==0
                        v7 = v7==0
                        v8 = v8==0
                        v9 = v9==0
                        A = (sum([(v2 and not v3), (v3 and not v4), (v4 and not v5), (v5 and not v6), (v6 and not v7), (v7 and not v8), (v8 and not v9), (v9 and not v2)]))
#                        self.time4.append(time.clock()-t0)
                        if (A == 1): 
                            marker[y][x] = 1
        return numpy.logical_and(img, numpy.logical_not(marker)).astype(numpy.uint8)
    def compute(self, img):
        img[numpy.where(img>0)] = 1
        prev = numpy.zeros_like(img, dtype = numpy.uint8)
        diff=None
    
        while (numpy.sum(diff) > 0 or diff==None):
            img = self.thinningIteration(img, 0)
            img = self.thinningIteration(img, 1)
            diff = cv2.absdiff(img, prev)
            prev = numpy.copy(img)
    
        return img.astype(numpy.uint8)*255
