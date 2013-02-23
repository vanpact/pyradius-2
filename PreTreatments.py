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
        Applier.__single.filtersToPreApply = None
        Applier.__single.filtersToPreApply = None
        Applier.__single.filtersToApply = None
        Applier.__single.filtersToApply = None
        Applier.__single.processedVideo = None
        Applier.__single.lastComputedFrame = None
        Applier.__single.wait=True
        Applier.__single.src = None
    
    @staticmethod
    def getInstance(src=None, nrChannel=1):
        if(Applier.__single is None):
            Applier.__single = Applier()
#            Applier.__single = Applier.__single
        Applier.__single.nrChannel=nrChannel
        Applier.__single.filtersToPreApply = []
        Applier.__single.filtersToPreApply = [ [] for i in range(nrChannel)]
        Applier.__single.filtersToApply = []
        Applier.__single.filtersToApply = [[] for i in range(nrChannel)]
        Applier.__single.processedVideo = []
        Applier.__single.lastComputedFrame = None
        Applier.__single.wait=True
        if(isinstance(src, Movie)):
            Applier.__single.src = src
        elif src is not None:
            raise TypeError("Src must be None or a Movie!") 
        return Applier.__single
        
    def toggle(self):
        """Toggle the applier state. if wait is true, the applier will pause the processing of the video."""
        self.wait = not self.wait
        
    def setSource(self, src=file):
        if(isinstance(src, Movie)):
            self.wait=True
            self.processedVideo = []
            self.src = src
        else:
            raise TypeError("Src must be none or a Movie!")
        
    def add(self, other, channel=0):
        if(isinstance(other, AbstractPreTreatment)):
            self.filtersToPreApply[channel].append(other)
            return self
        elif(isinstance(other, AbstractTreatment)):
            self.filtersToApply[channel].append(other)
            return self
        else:
            raise TypeError("Object send to the applier has to be a pretreatment.")
    
    def sub(self, other, channel=0):
        if(isinstance(other, AbstractPreTreatment)):
            self.filtersToPreApply[channel].reverse()
            self.filtersToPreApply[channel].remove(other)
            self.filtersToPreApply[channel].reverse()
        elif(isinstance(other, AbstractTreatment)):
            self.filtersToApply[channel].reverse()
            self.filtersToApply[channel].remove(other)
            self.filtersToApply[channel].reverse()
            return self
        else:
            raise TypeError("Object send to the applier has to be a pretreatment.")
        
    def empty(self):
        self.filterToPreApply = []
        
    def apply(self, img):
        imgToMix=[numpy.copy(img) for i in range(self.nrChannel)]
        for channel in range(self.nrChannel):
            imgPre = imgToMix[channel]
            for preTreatment in self.filtersToPreApply[channel]:
                imgPre = preTreatment.compute(imgPre)
            for treatment in self.filtersToApply[channel]:
                imgToMix[channel] = treatment.compute(imgPre, imgToMix[channel])
        for imgProcessed in imgToMix:
            img = numpy.maximum(img, imgProcessed)
#        img=imgPre
        return img
    
    def applyNext(self):
        self.src.readNextFrame()
        ndimg = self.src.rawBuffer
        if(len(ndimg.shape)>2):
            ndimg = ndimg[:, :, 0]*0.299+ndimg[:, :, 1]*0.587+ndimg[:, :, 2]*0.114
        ndimg = self.apply(ndimg)
        self.lastComputedFrame=ndimg
#        self.processedVideo.append(ndimg)
        self.frameComputed.emit()
        return QtMultimedia.QVideoFrame(ImageConverter.ndarrayToQimage(ndimg))
    
    def applyOne(self):
        self.applyNext()
        
    def applyAll(self):
        frameNb = self.src.getFrameNumber()
        for i in range(0, frameNb):
            self.applyNext()
            while(self.wait):
                self.msleep(50)
    
    def run(self):
        self.applyAll()
        
    def getLastComputedFrame(self):
        frameToSend = ImageConverter.ndarrayToQimage(self.lastComputedFrame).copy()#self.processedVideo[-1]).copy()
        return QtMultimedia.QVideoFrame(frameToSend)
    
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
    
    def __init__(self, ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0, ktype = numpy.float32, angleToProcess=None):
        super(GaborTreatment, self).__init__()
        self.angleToProcess = angleToProcess
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.ktype = ktype
        self.cpuNumber = max(multiprocessing.cpu_count()-2, 1)
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
        ymax = -ymax
        ymax = xmax
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
                kernel[ymax - y, xmax - x] = v
        return kernel
    
    def buildFilters(self):
        angles = 0
        if(self.angleToProcess is None):
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
#        dstSize=((img.shape[1]+1)/2, (img.shape[0]+1)/2)#Need to inverse width and height?
#        image = None
#        if(self.performance):
#            if(len(img.shape)>2):
#                image = numpy.ndarray((dstSize[1], dstSize[0], img.shape[2]))
#                for d in range(0, img.shape[2]): 
#                    image[:, :, d]= cv2.pyrDown(img[:, :, d], dstsize=dstSize)

        if(len(img.shape)==2):
            if(self.cpuNumber>1):
                img = self.processThreaded(numpy.copy(img))
            else:
                img = self.process(numpy.copy(img))
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
#        if(self.performance):
#            dstSize = (img.shape[1], img.shape[0])
##            if(len(img.shape)>2):
##                image = numpy.uint8(image)
##                for d in range(0, img.shape[2]):
##                    img[:, :, d] = cv2.pyrUp(image[:, :, d], dstsize=dstSize)
##            elif(len(img.shape)==2):
#            img = cv2.pyrUp(image, dstsize=dstSize)
##            else:
##                raise ValueError("The image must have 2 or 3 dimension.")
#        else:
#            img=image
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
        norm =  cv2.addWeighted( numpy.abs(grad_x), 0.5, numpy.abs(grad_y), 0.5, 0)
        return numpy.uint8((norm/numpy.max(norm))*255.0)
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
        img[img<self.threshold]=0
        img[img>=self.threshold]=255
        return img

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
           
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    image = QtGui.QImage('testGoodSizeGood.png')
    gaborProcessing = GaborTreatment(performance=True)
    cannyProcessing = CannyTreatment()
    img = ImageConverter.qimageToNdarray(image, True)
    
    t0 = time.clock()
    img = gaborProcessing.compute(img)
    print time.clock()-t0
    image = ImageConverter.ndarrayToQimage(img)
    image.save('Gabor.png')
    
    img = cannyProcessing.compute(img)
    image = ImageConverter.ndarrayToQimage(img)
    image.save('canny.png')