#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    .. module:: PreTreatments
        :platform: Unix, Windows
        :synopsis: module which provide the preTreatments for the video.
    .. moduleauthor:: Yves-Rémi Van Eycke <yveycke [at] ulb.ac.be>
"""


import cv2, numpy
from multiprocessing.pool import ThreadPool
import multiprocessing
import math

class AbstractPreTreatment(object):
    """Abstract class. All pretreatments inherit from this class"""
    def __init__(self):
        """
        Constructor
        """
        if(type(self) is AbstractPreTreatment):
            raise NotImplementedError('This class is abstract and cannot be instantiated') 
        super(AbstractPreTreatment, self).__init__()

    
class CannyTreatment(AbstractPreTreatment):
    """
    Canny edge detector
    """
    
    def __init__(self, minThreshold = 200, ratio = 1.2, kernelSize = 7, L2gradient=False):
        """    
        Constructor 
		
        :param minThreshold: The minimum threshold used by the Canny edge detector
        :type minThreshold: int
        :param ratio: use to know the maximal threshold used by the Canny edge detector : maxThreshold=ratio*minThreshold
        :type ratio: float
        :param kernelSize: The size of the kernel used for the gradient computation
        :type kernelSize: int
        :param L2Gradient: The method used to computer the gradient value. If true, results are more accurate but the method is slower.
        :type L2Gradient: bool
        """
        super(CannyTreatment, self).__init__()
        self.minThreshold = minThreshold
        self.ratio = ratio
        self.kernelSize = kernelSize
        self.L2gradient = L2gradient
        
    def compute(self, img):
        """        
        Process one image
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(len(img.shape)==2):
            img= cv2.GaussianBlur( img, (9, 9), sigmaX=1.0, sigmaY=1.0)
            edges = cv2.Canny(numpy.uint8(img), self.minThreshold, self.minThreshold*self.ratio, apertureSize=self.kernelSize, L2gradient=self.L2gradient)#self.kernelSize )
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
        return edges

class GaussianBlurTreatment(AbstractPreTreatment):
    """
    Gaussian blur
    """
    
    def __init__(self, kernelSize = (7, 7), Sigma = (1.0, 1.0)):
        """        
        Constructor
        
        :param kernelSize: The size of the kernel used.
        :type kernelSize: int  
        :param Sigma: The variance of the gaussian function.
        :type Sigma: float
        """
        super(GaussianBlurTreatment, self).__init__()
        self.kernelSize = kernelSize
        self.sigmaX = Sigma[0]
        self.sigmaY = Sigma[1]
        
    def compute(self, img):
        """        
        Process one image
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(len(img.shape)==2):
            return cv2.GaussianBlur( img, self.kernelSize, sigmaX=self.sigmaX, sigmaY=self.sigmaY)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
    
        
class GaborTreatment(AbstractPreTreatment):
    """The Gabor filter bank treatment"""
    filters = []
    
    def __init__(self, ksize = 31, sigma = 1.5, lambd = 15, gamma = 0.02, psi = 0, ktype = numpy.float32, angleToProcess=[]):
        """        
        Constructor
        
        :param ksize: The size of the kernel used.
        :type ksize: int  
        :param sigma: The variance.
        :type sigma: float
        :param lambd: The wavelength
        :type lambd: int  
        :param gamma: The excentricity
        :type gamma: float
        :param psi: the offset
        :type psi: int  
        :param ktype: The type of the kernel value
        :type ktype: Numpy type
        :param angleToProcess: The angles to process
        :type angleToProcess: list of float  
        """
        super(GaborTreatment, self).__init__()
        self.angleToProcess = angleToProcess
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.ktype = ktype
        self.cpuNumber = max(multiprocessing.cpu_count()-1, 1)
        if(self.cpuNumber>1):
            self.pool = ThreadPool(processes=self.cpuNumber)
        self.buildFilters()
       
    def getGaborKernel(self, theta): 
        """        
        Create a Gabor kernel with all the arguments given in the constructor. the Theta is the last missing argument
        
        :param theta: The orientation of the kernel
        :type theta: float
        """
        sigma_x = self.sigma
        sigma_y = self.sigma/self.gamma
        nstds = 3
        c = math.cos(theta)
        s = math.sin(theta)
        if( self.ksize > 0 ):
            xmax = self.ksize/2
        else:
            xmax = max(math.fabs(nstds*sigma_x*c), math.fabs(nstds*sigma_y*s))
        ymax = xmax
        ymin = -ymax
        xmin=-xmax
        kernel = numpy.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype = self.ktype)
        scale = 1.0
        ex = -0.5/(sigma_x*sigma_x)
        ey = -0.5/(sigma_y*sigma_y)
        cscale = numpy.pi*2/self.lambd
        
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                xr = x*c + y*s
                yr = -x*s + y*c
                v = scale*math.exp(ex*xr*xr + ey*yr*yr)*math.cos(cscale*xr + self.psi)
                kernel[ymax + y, xmax + x] = v
        return kernel
    
    def buildFilters(self):
        """                
        Build all the kernel for the Gabor bank filter. the orientation is the varying parameter
        """
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
        """        
        Process one image
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        accum = numpy.zeros_like(img)
        for kern, _ in self.filters:
            fimg = cv2.filter2D(img[:, :], cv2.CV_8U, kern)
            numpy.maximum(accum, fimg, accum)
            del(fimg)
        del(img)
        return accum

    def processThreaded(self, img):
        """        
        Process one image (multi-threaded version of process).
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        accum = numpy.zeros_like(img)
        accumLock = multiprocessing.Lock()
        def f(filt):
            kern, _ = filt
            fimg = cv2.filter2D(img[:, :], cv2.CV_8U, kern)
            with accumLock:
                idx = numpy.argmax(numpy.dstack((accum, fimg)),axis=2)
                accum[idx==1]=fimg[idx==1]
                del idx
            del fimg
            del kern
            del filt
        
        self.pool.map(f,self.filters)
        return accum
    
    def compute(self, img):
        """        
        Process one image by using process or processThreaded (choice is made automatically).
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(len(img.shape)==2):
            if(self.cpuNumber>1):
                img = self.processThreaded(img)
            else:
                img = self.process(img)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")
        return img

class ReduceSizeTreatment(AbstractPreTreatment):
    """Reduce the size of an image"""
    def __init__(self, dstSize=None):
        """        
        Constructor
        
        :param dstSize: The final size
        :type dstSize: Tuple of int
        """
        super(ReduceSizeTreatment, self).__init__()
        self.dstSize=dstSize
        
    def compute(self, img):
        """        
        Process one image
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(len(img.shape)==2):
                return cv2.pyrDown(img, dstsize=self.dstSize)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class IncreaseSizeTreatment(AbstractPreTreatment):
    """Increase the size of an image"""
    def __init__(self, dstSize=None):
        """        
        Constructor
        
        :param dstSize: The final size
        :type dstSize: Tuple of int
        """
        super(IncreaseSizeTreatment, self).__init__()
        self.dstSize=dstSize
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(len(img.shape)==2):
                return numpy.uint8(cv2.pyrUp(img, dstsize=self.dstSize))
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class changeContrastTreatment(AbstractPreTreatment):
    """Change the contrast of the image"""
    def __init__(self, value=1.0):
        """        
        Constructor
        
        :param value: Contrast multiplier
        :type value: float
        """
        super(changeContrastTreatment, self).__init__()
        self.value=value
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        image = numpy.zeros(img.shape, dtype = numpy.uint32)
        if(len(img.shape)==2):
                image = numpy.uint32(img)*self.value
                image[image >255] = 255
                return numpy.uint8(image)
        else:
            raise ValueError("The image must have 2 dimension(gray scale).")

class cropTreatment(AbstractPreTreatment):
    """Crop the image"""
    def __init__(self, roi = None):
        """        
        Constructor
        
        :param roi: Coordinates of the region of interest
        :type roi: Tuple of Tuple of int
        """
        super(cropTreatment, self).__init__()        
        self.roi = roi
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(self.roi is not None and self.roi[0][0]>=0 and self.roi[0][1]>=0 and self.roi[0][0]<self.roi[1][0] and self.roi[0][1]<self.roi[1][1] and self.roi[1][0]<img.shape[0] and self.roi[1][1]<img.shape[1]):
            return img[self.roi[0][1]:self.roi[1][1],self.roi[0][0]:self.roi[1][0]]
        else: 
            raise ValueError("Incorrect size for the region of interest when cropping.")    

class rotateTreatment(AbstractPreTreatment):
    """Rotate the image"""
    def __init__(self, angle = 0):
        """        
        Constructor
        
        :param angle: The angle of rotation
        :type value: float
        """
        super(rotateTreatment, self).__init__()
        self.setAngle(angle)
    
    def setAngle(self, angle):
        """        
        Change the angle
        
        :param angle: The angle of rotation
        :type angle: float
        """
        self.angle=angle
        self.alpha = numpy.cos(-self.angle)
        self.beta = numpy.sin(-self.angle)
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        center = (img.shape[0]/2.0, img.shape[1]/2.0)
        M = cv2.getRotationMatrix2D(center, numpy.degrees(self.angle), 1.0)
        sizemaxX = numpy.int(numpy.abs(self.alpha*img.shape[1]))
        sizemaxY = numpy.int(numpy.abs(self.beta*img.shape[1]+self.alpha*img.shape[0]))
        return cv2.warpAffine(img, M, (sizemaxX, sizemaxY), borderMode=cv2.BORDER_WRAP).copy()
        
class LaplacianTreatment(AbstractPreTreatment):
    """Perform filtering using a Laplacian of Gaussian"""
    def __init__(self, kernelSize=3, scale=1, delta=0):
        """        
        Constructor
        
        :param kernelSize: The kernel size
        :type kernelSize: int
        :param scale: The scale factor computer for the laplacian value
        :type scale: float
        :param delta: value added to the results
        :type delta: int
        """
        super(LaplacianTreatment, self).__init__()  
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        result = cv2.Laplacian( numpy.uint8(img), ddepth=cv2.CV_32F, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        result[result<0]=0
        return numpy.uint8((result/numpy.max(result))*255.0)
    
class SobelTreatment(AbstractPreTreatment):
    """Compute the Sobel derivative of an image"""
    def __init__(self, dx=1, dy=1, kernelSize=7, scale=1, delta=0):
        """        
        Constructor
        
        :param dx: The order of the x derivative
        :type kernelSize: int
        :param dy: The order of the y derivative
        :type dy: float
        :param kernelSize: The kernel size
        :type kernelSize: int
        :param scale: The scale factor for the derivative values
        :type scale: float
        :param delta: value added to the results
        :type delta: int
        """
        super(SobelTreatment, self).__init__()  
        self.dx=dx
        self.dy=dy
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        result = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=self.dx, dy=self.dy, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        result[result<0] = 0
        result = result - numpy.min(result)
        del(img)
        return numpy.uint8((result/numpy.max(result))*255.0)
        return result
    
#     def compute2(self, img):
#         grad_x = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7, scale=1, delta=0)
#         
#         grad_y = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=0, dy=1, ksize=7, scale=1, delta=0)
#         norm =  numpy.sqrt(numpy.add(numpy.square(numpy.asarray(grad_x, dtype=numpy.float)), numpy.square(numpy.asarray(grad_y, dtype=numpy.float))))#cv2.addWeighted( numpy.abs(grad_x), 0.5, numpy.abs(grad_y), 0.5, 0)
#         return numpy.asarray((norm/numpy.max(norm))*255.0, dtype=numpy.uint8)

class DOHTreatment(AbstractPreTreatment):
    """Compute the determinant of the Hessian matrix"""
    def __init__(self, kernelSize=7, scale=1, delta=0):
        """        
        Constructor
        
        :param kernelSize: The kernel size
        :type kernelSize: int
        :param scale: The scale factor for the derivative values
        :type scale: float
        :param delta: value added to the results
        :type delta: int
        """
        super(DOHTreatment, self).__init__()  
        self.kernelSize=kernelSize
        self.scale=scale
        self.delta=delta
    
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        resultxy = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=1, dy=1, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        resultxx = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=2, dy=0, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        resultyy = cv2.Sobel( numpy.uint8(img), ddepth=cv2.CV_32F, dx=0, dy=2, ksize=self.kernelSize, scale=self.scale, delta=self.delta);
        result = resultxx*resultyy-resultxy*resultxy
        
        result[not numpy.all((result>0, resultxx<=0))]=0
        result[numpy.all((result>0, resultxx<=0))]=255
        return numpy.uint8((result/numpy.max(result))*255.0)
    
class ThresholdTreatment(AbstractPreTreatment):
    """"Perform Thesholding"""
    def __init__(self, threshold=-1):
        """        
        Constructor
        
        :param threshold: The threshold. -1 means automatic threshold.
        :type threshold: int
        """
        super(ThresholdTreatment, self).__init__()
        self.threshold = threshold
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        if(self.threshold<-1):
            self.threshold=numpy.mean(img[img>=numpy.max(self.threshold, 0)])
        elif(self.threshold<0):
            self.threshold=numpy.mean(img[img>numpy.max(numpy.min(img), 0)])
        img[img<self.threshold]=0
        img[img>=self.threshold]=255
        return img
    
class addHlineTreatment(AbstractPreTreatment):
    """add horizontal white line on the image"""
    def __init__(self, thickness = 1, lineDistance=10):
        """        
        Constructor
        
        :param thickness: The line thickness
        :type thickness: int
        :param lineDistance: The distance between the lines
        :type lineDistance: int
        """
        super(addHlineTreatment, self).__init__()
        self.thickness = thickness
        self.lineDistance = numpy.int(lineDistance)
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        x = img.shape[1]
        y = img.shape[0]
        for i in numpy.arange(self.lineDistance-1,y,self.lineDistance):
            cv2.rectangle(img, (0, i), (x, i+self.thickness), (0, 0, 0), thickness=-1)
        return img.copy()
        
class DilationTreatment(AbstractPreTreatment):
    """Perform morphological dilation"""
    def __init__(self, morphology=cv2.MORPH_RECT, size=(3, 3)):
        """        
        Constructor
        
        :param morphology: The shape of the mask
        :type morphology: int
        :param size: The size of the shape
        :type size: int
        """
        super(DilationTreatment, self).__init__()
        self.morphology=morphology
        self.size=size
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        elem = cv2.getStructuringElement( self.morphology, self.size )
        return cv2.dilate(img, elem)

class erosionTreatment(AbstractPreTreatment):
    """Perform morphological erosion"""
    def __init__(self, morphology=cv2.MORPH_RECT, size=(3, 3)):
        """        
        Constructor
        
        :param morphology: The shape of the mask
        :type morphology: int
        :param size: The size of the shape
        :type size: int
        """
        super(erosionTreatment, self).__init__()
        self.morphology=morphology
        self.size=size
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
        elem = cv2.getStructuringElement( self.morphology, self.size )
        return cv2.erode(img, elem)

class SkeletonTreatment(AbstractPreTreatment):
    """Compute morphological skeleton of blobs in an image"""
    def __init__(self, size=(3, 3)):
        """        
        Constructor
        
        :param size: The size of the cross shaped mask used in the method
        :type size: int
        """
        super(SkeletonTreatment, self).__init__()
        self.size=size
        
    def compute(self, img):
        """        
        Process one image.
        
        :param img: The image to process
        :type img: Numpy array    
        :return: The processed image
        :rtype: Numpy array
        """
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
        return skel.astype(numpy.uint8)*255
    
# class ThinningTreatment(AbstractTreatment):
#     def __init__(self, size=(3, 3)):
#         super(ThinningTreatment, self).__init__()
#         self.size=size
#     def thinningIteration(self, img, itera):
#         marker = numpy.zeros_like(img, dtype = numpy.uint8)
#         for y in numpy.arange(1, img.shape[0]-2, 1):
#             for x in numpy.arange(1, img.shape[1]-2, 1):
#                 v2 = img[y-1, x]
#                 v3 = img[y-1, x+1]
#                 v4 = img[y, x+1]
#                 v5 = img[y+1, x+1]
#                 v6 = img[y+1, x]
#                 v7 = img[y+1, x-1]
#                 v8 = img[y, x-1]
#                 v9 = img[y-1, x-1]
#                 B = v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
#                 if(B >= 2 and B <= 6):
#                     if(itera == 0):
#                         m1 = (v2 * v4 * v6)
#                         m2 = (v4 * v6 * v8)
#                     else :
#                         m1 = (v2 * v4 * v8)
#                         m2 = (v2 * v6 * v8)
#                     if( m1 == 0 and m2 == 0):
#                         v2 = v2==0
#                         v3 = v3==0
#                         v4 = v4==0
#                         v5 = v5==0
#                         v6 = v6==0
#                         v7 = v7==0
#                         v8 = v8==0
#                         v9 = v9==0
#                         A = (sum([(v2 and not v3), (v3 and not v4), (v4 and not v5), (v5 and not v6), (v6 and not v7), (v7 and not v8), (v8 and not v9), (v9 and not v2)]))
#                         if (A == 1): 
#                             marker[y][x] = 1
#         return numpy.logical_and(img, numpy.logical_not(marker)).astype(numpy.uint8)
#     def compute(self, img):
#         img[numpy.where(img>0)] = 1
#         prev = numpy.zeros_like(img, dtype = numpy.uint8)
#         diff=None
#     
#         while (numpy.sum(diff) > 0 or diff==None):
#             img = self.thinningIteration(img, 0)
#             img = self.thinningIteration(img, 1)
#             diff = cv2.absdiff(img, prev)
#             prev = numpy.copy(img)
#     
#         return img.astype(numpy.uint8)*255

